import functools
import operator
from typing import cast, Dict, List, Optional, Sequence, Tuple
import torch
import torch.distributed as dist
import torch.distributed._tensor.api as dtensor
import torch.distributed._tensor.random as random
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.placement_types import DTensorSpec, Replicate, TensorMeta
from torch.distributed._tensor.random import is_rng_supported_mesh
from torch.distributed._tensor.redistribute import redistribute_local_tensor
from torch.distributed._tensor.sharding_prop import ShardingPropagator
from torch.distributed._tensor.tp_conv import (
from torch.distributed.device_mesh import DeviceMesh
@staticmethod
def redistribute_local_args(op_info: OpInfo, suggested_input_schema: OpSchema) -> None:
    if op_info.args_tree_spec is not None:
        flatten_args_schema_to_reshard = tuple(pytree.tree_leaves(suggested_input_schema.args_schema))
    else:
        flatten_args_schema_to_reshard = suggested_input_schema.args_schema
    new_local_args: List[object] = []
    for i, arg_spec in enumerate(op_info.flat_args_schema):
        reshard_arg_spec = flatten_args_schema_to_reshard[i]
        if isinstance(arg_spec, DTensorSpec):
            local_tensor = cast(torch.Tensor, op_info.local_args[i])
            if arg_spec != reshard_arg_spec:
                resharded_local_tensor = redistribute_local_tensor(local_tensor, arg_spec, reshard_arg_spec)
                new_local_args.append(resharded_local_tensor)
            else:
                new_local_args.append(local_tensor)
        else:
            new_local_args.append(reshard_arg_spec)
    op_info.local_args = tuple(new_local_args)