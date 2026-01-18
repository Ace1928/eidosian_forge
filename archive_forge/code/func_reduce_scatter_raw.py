from typing import Optional
import torch
from torch import Tensor
from torch.distributed import ProcessGroup
def reduce_scatter_raw(input_: Tensor, process_group: ProcessGroup, async_op: bool=False):
    world_size = torch.distributed.get_world_size(process_group)
    assert input_.shape[0] % world_size == 0
    output = torch.empty(input_.shape[0] // world_size, *input_.shape[1:], dtype=input_.dtype, device=input_.device)
    handle = torch.distributed.reduce_scatter_tensor(output, input_.contiguous(), group=process_group, async_op=async_op)
    return (output, handle)