from contextlib import contextmanager
from typing import Optional
import torch
import torch.distributed as dist
from vllm.logger import init_logger
from vllm.model_executor.parallel_utils.parallel_state import (
def init_custom_ar() -> None:
    global _CA_HANDLE
    if _CA_HANDLE is not None:
        return
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return
    if world_size not in _SUPPORTED_WORLD_SIZES:
        logger.warn('Custom allreduce is disabled due to an unsupported world size: %d. Supported world sizes: %s. To silence this warning, specifydisable_custom_all_reduce=True explicitly.', world_size, str(_SUPPORTED_WORLD_SIZES))
        return
    if not _can_p2p(rank, world_size):
        logger.warn('Custom allreduce is disabled because your platform lacks GPU P2P capability. To silence this warning, specifydisable_custom_all_reduce=True explicitly.')
        return
    _CA_HANDLE = CustomAllreduce(rank, world_size)