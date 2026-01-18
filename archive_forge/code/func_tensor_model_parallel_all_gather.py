from collections import namedtuple
from typing import Any, Dict, List, Optional, Union
import torch
from torch.distributed import ProcessGroup
from vllm.model_executor.parallel_utils import cupy_utils
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.parallel_utils.custom_all_reduce import custom_all_reduce
def tensor_model_parallel_all_gather(input_: torch.Tensor, dim: int=-1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return input_
    assert -input_.dim() <= dim < input_.dim(), f'Invalid dim ({dim}) for input tensor with shape {input_.size()}'
    if dim < 0:
        dim += input_.dim()
    input_size = input_.size()
    output_tensor = torch.empty((world_size,) + input_size, dtype=input_.dtype, device=input_.device)
    torch.distributed.all_gather_into_tensor(output_tensor, input_, group=get_tensor_model_parallel_group())
    output_tensor = output_tensor.movedim(0, dim)
    output_tensor = output_tensor.reshape(input_size[:dim] + (world_size * input_size[dim],) + input_size[dim + 1:])
    return output_tensor