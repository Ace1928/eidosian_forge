from collections import namedtuple
from typing import Any, Dict, List, Optional, Union
import torch
from torch.distributed import ProcessGroup
from vllm.model_executor.parallel_utils import cupy_utils
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.parallel_utils.custom_all_reduce import custom_all_reduce
def tensor_model_parallel_gather(input_: torch.Tensor, dst: int=0, dim: int=-1) -> torch.Tensor:
    """Gather the input tensor across model parallel group.

    NOTE: We assume that the input tensor is on the same device across
    all the ranks.
    """
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return input_
    assert -input_.dim() <= dim < input_.dim(), f'Invalid dim ({dim}) for input tensor with shape {input_.size()}'
    if dim < 0:
        dim += input_.dim()
    if get_tensor_model_parallel_rank() == dst:
        gather_list = [torch.empty_like(input_) for _ in range(world_size)]
    else:
        gather_list = None
    torch.distributed.gather(input_, gather_list, dst=dst, group=get_tensor_model_parallel_group())
    if get_tensor_model_parallel_rank() == dst:
        output_tensor = torch.cat(gather_list, dim=dim)
    else:
        output_tensor = None
    return output_tensor