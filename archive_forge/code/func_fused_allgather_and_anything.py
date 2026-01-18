import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union, overload
import torch
import torch.distributed as dist
import torch.multiprocessing.reductions
from .. import _is_triton_available
from .common import BaseOperator, get_xformers_operator, register_operator
from .ipc import init_ipc
def fused_allgather_and_anything(scattered_inputs: List[torch.Tensor], my_matmul: Callable[[List[torch.Tensor], int, Callable[[], torch.cuda.Stream]], None], *, group: dist.ProcessGroup, num_stripes: int=1, timeout_s: int=60 * 60, **private_args_DO_NOT_USE) -> None:
    world_size = group.size()
    if len(scattered_inputs) == 0:
        for src_rank in range(world_size):
            my_matmul([], src_rank, _default_stream_factory)
        return
    assert all((si.is_contiguous() for si in scattered_inputs))
    assert all((si.device == scattered_inputs[0].device for si in scattered_inputs))
    assert all((si.dtype == scattered_inputs[0].dtype for si in scattered_inputs))
    gathered_input_shapes = [(world_size,) + si.shape for si in scattered_inputs]
    obj = _lazy_init(scattered_inputs[0].device, scattered_inputs[0].dtype, group, num_stripes)
    if world_size == 1:
        my_matmul(scattered_inputs, 0, _default_stream_factory)
    elif obj is None:
        gathered_inputs = [si.new_empty(gis) for si, gis in zip(scattered_inputs, gathered_input_shapes)]
        for si, gi in zip(scattered_inputs, gathered_inputs):
            dist.all_gather_into_tensor(output_tensor=gi, input_tensor=si, group=group)
        for src_rank in range(world_size):
            my_matmul([gi[src_rank] for gi in gathered_inputs], src_rank, _default_stream_factory)
    else:
        assert scattered_inputs[0].device == obj.my_device
        assert scattered_inputs[0].dtype == obj.dtype
        assert obj.num_stripes == num_stripes
        obj.allgather_and_linear(scattered_inputs, my_matmul, timeout_s=timeout_s, _wait=private_args_DO_NOT_USE.get('_wait', True), _memcpy=private_args_DO_NOT_USE.get('_memcpy', True), _triton=private_args_DO_NOT_USE.get('_triton', True), _is_regular_matmul=private_args_DO_NOT_USE.get('_is_regular_matmul', False), _extra_triton_args=private_args_DO_NOT_USE.get('_extra_triton_args', {}))