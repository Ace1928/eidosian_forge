import logging
import math
from typing import List, Optional
import torch
import torch.distributed._tensor.placement_types as placement_types
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh
from torch.distributed.distributed_c10d import (
def mesh_scatter(output: torch.Tensor, scatter_list: List[torch.Tensor], mesh: DeviceMesh, mesh_dim: int=0, async_op: bool=False) -> Optional[Work]:
    """
    scatter a list of tensors to a device mesh dimension. We by default
    use the first rank of the mesh dimension as the source of truth, i.e
    for a 2d mesh [[0, 1], [2, 3]], if we scatter on mesh_dim = 1, we will
    scatter the tensor list on rank 0 to rank 0/1, and tensor list on rank
    2 to rank 2/3.

    Args:
        output (torch.Tensor): the tensor to receive the scattered list.
        scatter_list (List[torch.Tensor]): the tensor list to be scattered.
        mesh_dim (int, optional): indicate which mesh dimension we want
            to scatter on, we by default choose the first rank on the
            mesh dimension as source of truth.

    Returns:
        A :class:`Work` object
    """
    if output.is_meta:
        return None
    dim_group = mesh.get_group(mesh_dim)
    assert isinstance(dim_group, ProcessGroup)
    src_for_dim = 0
    if dim_group is not GroupMember.WORLD:
        src_for_dim = get_global_rank(dim_group, 0)
    if src_for_dim == get_rank():
        fut = scatter(output, scatter_list=scatter_list, src=src_for_dim, group=dim_group, async_op=async_op)
    else:
        fut = scatter(output, scatter_list=None, src=src_for_dim, group=dim_group, async_op=async_op)
    return fut