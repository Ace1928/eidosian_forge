from dataclasses import dataclass
from typing import Callable, cast, Dict, Iterable, Optional, Sequence, Set, Tuple, Union
import torch
from torch import Tensor
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.api import Shard
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.utils import (
from torch.distributed._tensor.placement_types import DTensorSpec, Placement, Replicate
from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing
def propagate_shape_and_sharding(in_shard: Sequence[Placement], local_in_shape: Shape, rule: DimMap, mesh_sizes: Shape) -> Tuple[Shape, Optional[Sequence[Placement]], torch.Tensor]:
    """
    Determine output sharding and tensor shape based on given global tensor shape and input sharding.

    Takes as input the global shape of the tensor, and the input sharding,
    and produce corresponding output sharding and shape of the output tensor.

    Sharding propagation follows mapped dimensions:
    - An output dimension that maps directly to an input dimension is sharded equally
    - An output dimension that is a flattened set of input dimensions can only be
      sharded if only the leftmost flattened dimension is sharded.
    - An output dimension that is a split of the input dimension can only be sharded
      if the leftmost split size is divisible by the mesh dimension
    """
    assert len(in_shard) == len(mesh_sizes)
    sharded_in_dims: Set[int] = {s.dim for s in in_shard if isinstance(s, Shard)}
    shardable_dims: torch.Tensor = torch.ones((len(local_in_shape), len(mesh_sizes)), dtype=torch.bool)
    seen_input_dims: Set[int] = set()

    def collect_used_inputs(cmd: DimSpec) -> None:
        if isinstance(cmd, InputDim):
            seen_input_dims.add(cmd.input_dim)
        for inp in cmd.inputs():
            collect_used_inputs(inp)
    for cmd in rule:
        collect_used_inputs(cmd)
    for dim in range(len(local_in_shape)):
        shardable_dims[dim, :] = dim in seen_input_dims

    def get_dim_size(cmd: DimSpec) -> Tuple[int, Optional[InputDim]]:
        if isinstance(cmd, InputDim):
            seen_input_dims.add(cmd.input_dim)
            return (local_in_shape[cmd.input_dim], cmd if cmd.input_dim in sharded_in_dims else None)
        elif isinstance(cmd, Flatten):
            for dim in cmd.input_dims[1:]:
                if isinstance(dim, InputDim):
                    shardable_dims[dim.input_dim, :] = False
            dim0 = cmd.input_dims[0]
            return (prod((get_dim_size(a)[0] for a in cmd.input_dims)), dim0 if isinstance(dim0, InputDim) and dim0.input_dim in sharded_in_dims else None)
        elif isinstance(cmd, Split):
            _, in_dim = get_dim_size(cmd.input_dim)
            out_size = cmd.group_shape[cmd.split_id]
            if cmd.split_id == 0 and in_dim is not None:
                for mesh_dim, mesh_dim_size in enumerate(mesh_sizes):
                    shardable_dims[in_dim.input_dim, mesh_dim] = out_size % mesh_dim_size == 0
                submesh_size = 1
                for size, shard in zip(mesh_sizes, in_shard):
                    if isinstance(shard, Shard) and shard.dim == in_dim:
                        submesh_size *= size
                assert out_size % submesh_size == 0, f'Resulting dimension size {out_size} is not divisible by its mesh dimension {submesh_size}.'
            return (out_size, in_dim if cmd.split_id == 0 else None)
        elif isinstance(cmd, Singleton):
            return (1, None)
        elif isinstance(cmd, Broadcast):
            return (cmd.dim_size, None)
        elif isinstance(cmd, NewDim):
            return (cmd.size, None)
        elif isinstance(cmd, Repeat):
            size, in_dim = get_dim_size(cmd.input_dim)
            if in_dim is not None:
                shardable_dims[in_dim.input_dim, :] = False
            return (size * cmd.times, None)
        else:
            raise RuntimeError(f'cmd not found: {cmd}, in rule: {rule}')
    dim_map = {}
    out_shape = []
    for dim, cmd in enumerate(rule):
        out_size, in_dim = get_dim_size(cmd)
        out_shape.append(out_size)
        if in_dim is not None:
            dim_map[in_dim.input_dim] = dim
    needs_reshard = any((isinstance(placement, Shard) and (not shardable_dims[placement.dim][mesh_dim]) for mesh_dim, placement in enumerate(in_shard)))
    output_placements = None if needs_reshard else [Shard(dim_map[s.dim]) if isinstance(s, Shard) else s for s in in_shard]
    return (tuple(out_shape), output_placements, shardable_dims)