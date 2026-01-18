from contextlib import contextmanager
from typing import Optional
import torch
import torch.distributed as dist
from vllm.logger import init_logger
from vllm.model_executor.parallel_utils.parallel_state import (
def custom_all_reduce(input: torch.Tensor) -> Optional[torch.Tensor]:
    ca_handle = get_handle()
    if ca_handle is None:
        return
    if is_capturing():
        if torch.cuda.is_current_stream_capturing():
            if ca_handle.should_custom_ar(input):
                return ca_handle.all_reduce_reg(input)
        elif ca_handle.should_custom_ar(input):
            return torch.empty_like(input)
    elif ca_handle.should_custom_ar(input):
        return ca_handle.all_reduce_unreg(input)