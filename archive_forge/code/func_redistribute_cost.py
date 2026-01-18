import logging
import math
from typing import List, Optional
import torch
import torch.distributed._tensor.placement_types as placement_types
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh
from torch.distributed.distributed_c10d import (
def redistribute_cost(current_spec: 'placement_types.DTensorSpec', target_spec: 'placement_types.DTensorSpec') -> float:
    """
    This function returns the cost of redistribute from current to target DTensorSpec.

    NOTE:
    1. Only consider communication cost here, since computation costs for redistribute
       are quite trival (i.e. we only need to narrow or simple division)
    2. Only consider redistribute cost on same mesh, cross mesh communication cost is
       not quite needed for operator strategy estimation/selection.
    """
    if current_spec.mesh != target_spec.mesh:
        return float('inf')
    if current_spec.is_replicated():
        return 0.0
    mesh = current_spec.mesh
    cost = 0.0
    comm_bytes = spec_to_bytes(current_spec) / current_spec.num_shards
    for i, (current, target) in enumerate(zip(current_spec.placements, target_spec.placements)):
        if current == target:
            continue
        if current.is_shard() and target.is_replicate():
            comm_bytes *= mesh.size(i)
            cost += allgather_cost(comm_bytes, current_spec.mesh, i)
        elif current.is_shard() and target.is_shard():
            cost += allgather_cost(comm_bytes, current_spec.mesh, i) + 1.0
        elif current.is_partial() and target.is_replicate():
            cost += allreduce_cost(comm_bytes, current_spec.mesh, i)
        elif current.is_partial() and target.is_shard():
            cost += reduce_scatter_cost(comm_bytes, current_spec.mesh, i)
            comm_bytes /= mesh.size(i)
        elif current.is_shard() and target.is_partial():
            return float('inf')
    return cost