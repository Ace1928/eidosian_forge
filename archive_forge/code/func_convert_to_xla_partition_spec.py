import logging
import os
from functools import wraps
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from torch.distributed._tensor.placement_types import Placement, Replicate
from torch.distributed.device_mesh import DeviceMesh
@with_xla
def convert_to_xla_partition_spec(tensor: torch.Tensor, placements: Sequence[Placement]) -> Tuple[Union[Tuple, int, None]]:
    """
    Convert DTensor `placements` to XLAShardedTensor `partitoin_spec`.
    This supports Shard and Replicate Placement types.

    Example:
      ```
      # Mesh partitioning, 1/4-th of the input with replicated overlaps.
      # The first input tensor dimension is sharded across the second mesh
      # dimension, and the rest is replicated over the first mesh dimension.
      t = torch.randn(4, 8, 8)
      dt_mesh = DeviceMesh("xla", torch.arange(8).reshape(2,4))
      placements = [Replicate(), Shard(0)]
      my_dtensor = distribute_tensor(t, dt_mesh, placements)

      # `placements = [Replicate(), Shard(0)]` describes sharding per mesh dim,
      # and this is equivalent to `partition_spec = (1, None, None)` which is
      # sharding per input tensor dimension.
      partition_spec = convert_to_xla_partition_spec(t, placements)
      >> (1, None, None)
      ```
    """
    sharding_spec = [None] * len(tensor.shape)
    for mesh_idx, spec in enumerate(placements):
        if spec.is_shard():
            tensor_idx = spec.dim
            sharding_spec[tensor_idx] = mesh_idx
        elif spec.is_replicate():
            continue
        else:
            raise ValueError(f'Unsupported placement type: {type(spec).__name__}')
    return tuple(sharding_spec)