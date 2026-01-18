import gc
import os
from typing import Dict, List, Tuple, Set, Optional
import torch
import torch.distributed
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
from vllm.model_executor import set_random_seed
from vllm.model_executor.parallel_utils import cupy_utils
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.parallel_utils.custom_all_reduce import init_custom_ar
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.model_runner import ModelRunner
from vllm.lora.request import LoRARequest
from vllm.utils import is_hip
def init_distributed_environment(parallel_config: ParallelConfig, rank: int, cupy_port: Optional[int], distributed_init_method: Optional[str]=None) -> None:
    """Initialize the distributed environment."""
    if torch.distributed.is_initialized():
        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != parallel_config.world_size:
            raise RuntimeError(f'torch.distributed is already initialized but the torch world size does not match parallel_config.world_size ({torch_world_size} vs. {parallel_config.world_size}).')
    elif not distributed_init_method:
        raise ValueError('distributed_init_method must be set if torch.distributed is not already initialized')
    else:
        torch.distributed.init_process_group(backend='nccl', world_size=parallel_config.world_size, rank=rank, init_method=distributed_init_method)
    if cupy_utils.is_initialized():
        cupy_world_size = cupy_utils.get_world_size()
        if cupy_world_size != parallel_config.world_size:
            raise RuntimeError(f'cupy.distributed is already initialized but the cupy world size does not match parallel_config.world_size ({cupy_world_size} vs. {parallel_config.world_size}).')
    elif parallel_config.world_size > 1 and cupy_port is not None and (not is_hip()):
        cupy_utils.init_process_group(world_size=parallel_config.world_size, rank=rank, host='localhost', port=cupy_port)
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    if cupy_utils.is_initialized():
        cupy_utils.all_reduce(torch.zeros(1).cuda())
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size, parallel_config.pipeline_parallel_size)
    if not parallel_config.disable_custom_all_reduce:
        init_custom_ar()