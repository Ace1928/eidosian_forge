from typing import Callable, Dict, List, Set
import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch import Tensor
from torch.distributed._tensor import DeviceMesh, Replicate, Shard
from torch.distributed._tensor.ops.view_ops import (
from torch.distributed._tensor.placement_types import _Partial, DTensorSpec
def compute_act_spec(self, node: fx.Node, mesh: DeviceMesh) -> DTensorSpec:
    """Compute the batch dimension for the current node, then generate the sharding spec that shards on the batch dimension."""
    node_batch_dim = self.compute_batch_dim(node)
    if node_batch_dim == -1:
        act_spec = DTensorSpec(mesh=mesh, placements=(Replicate(),))
    elif node_batch_dim == -2:
        act_spec = DTensorSpec(mesh=mesh, placements=(_Partial(),))
    else:
        act_spec = DTensorSpec(mesh=mesh, placements=(Shard(node_batch_dim),))
    return act_spec