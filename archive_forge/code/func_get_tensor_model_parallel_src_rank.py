import contextlib
import torch
from vllm.model_executor.parallel_utils import cupy_utils
def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return global_rank // local_world_size * local_world_size