import logging
import math
from typing import List, Optional
import torch
import torch.distributed._tensor.placement_types as placement_types
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh
from torch.distributed.distributed_c10d import (
def mesh_all_to_all(output_tensor_list: List[torch.Tensor], input_tensor_list: List[torch.Tensor], mesh: DeviceMesh, mesh_dim: int=0, async_op: bool=False) -> Optional[Work]:
    dim_group = mesh.get_group(mesh_dim)
    assert isinstance(dim_group, ProcessGroup)
    work = None
    if mesh.device_type == 'cpu':
        logger.warning('ProcessGroupGloo does not support all_to_all, falling back with scatters!')
        dim_group_size = get_world_size(dim_group)
        for i in range(dim_group_size):
            src_for_dim = i
            if dim_group is not GroupMember.WORLD:
                src_for_dim = get_global_rank(dim_group, i)
            work = scatter(output_tensor_list[i], input_tensor_list if mesh.get_rank() == src_for_dim else [], group=dim_group, src=src_for_dim, async_op=async_op)
    else:
        work = all_to_all(output_tensor_list, input_tensor_list, dim_group, async_op=async_op)
    return work