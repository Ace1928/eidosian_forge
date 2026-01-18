import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union, overload
import torch
import torch.distributed as dist
import torch.multiprocessing.reductions
from .. import _is_triton_available
from .common import BaseOperator, get_xformers_operator, register_operator
from .ipc import init_ipc
def fused_anything_and_reducescatter(my_matmul: Callable[[List[torch.Tensor], int, Callable[[], torch.cuda.Stream]], None], scattered_outputs: List[torch.Tensor], *, group: dist.ProcessGroup, num_stripes: int=1, timeout_s: int=60 * 60, **private_args_DO_NOT_USE) -> None:
    world_size = group.size()
    if len(scattered_outputs) == 0:
        for dst_rank in range(world_size):
            my_matmul([], dst_rank, _default_stream_factory)
        return
    assert all((so.is_contiguous() for so in scattered_outputs))
    assert all((so.device == scattered_outputs[0].device for so in scattered_outputs))
    assert all((so.dtype == scattered_outputs[0].dtype for so in scattered_outputs))
    gathered_output_shapes = [(world_size,) + so.shape for so in scattered_outputs]
    obj = _lazy_init(scattered_outputs[0].device, scattered_outputs[0].dtype, group, num_stripes)
    if world_size == 1:
        my_matmul(scattered_outputs, 0, _default_stream_factory)
    elif obj is None:
        gathered_outputs = [so.new_empty(gos) for so, gos in zip(scattered_outputs, gathered_output_shapes)]
        for dst_rank in range(world_size):
            my_matmul([go[dst_rank] for go in gathered_outputs], dst_rank, _default_stream_factory)
        for go, so in zip(gathered_outputs, scattered_outputs):
            dist.reduce_scatter_tensor(output=so, input=go, group=group)
    else:
        assert scattered_outputs[0].device == obj.my_device
        assert scattered_outputs[0].dtype == obj.dtype
        assert obj.num_stripes == num_stripes
        gathered_outputs = [scattered_outputs[0].new_empty(gos) for gos in gathered_output_shapes]
        obj.linear_and_reducescatter(my_matmul, gathered_outputs, scattered_outputs, timeout_s=timeout_s, _wait=private_args_DO_NOT_USE.get('_wait', True), _memcpy=private_args_DO_NOT_USE.get('_memcpy', True), _triton=private_args_DO_NOT_USE.get('_triton', True), _is_regular_matmul=private_args_DO_NOT_USE.get('_is_regular_matmul', False), _extra_triton_args=private_args_DO_NOT_USE.get('_extra_triton_args', {}))