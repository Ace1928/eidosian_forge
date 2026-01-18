from typing import Optional
import torch
from torch import Tensor
from torch.distributed import ProcessGroup
def all_reduce_raw(input_: Tensor, process_group: ProcessGroup, async_op: bool=False):
    input_ = input_.contiguous()
    handle = torch.distributed.all_reduce(input_, group=process_group, async_op=async_op)
    return (input_, handle)