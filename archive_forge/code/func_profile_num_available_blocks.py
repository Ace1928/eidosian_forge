from typing import Dict, List, Optional, Tuple
import torch
import torch.distributed
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
from vllm.model_executor import set_random_seed
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.model_runner import ModelRunner
@torch.inference_mode()
def profile_num_available_blocks(self, block_size: int=128, gpu_memory_utilization: float=0.9, cpu_swap_space: int=0, cache_dtype: str='float16') -> Tuple[int, int]:
    """Simply returns max_num_seqs as num_gpu_blocks, 0 as num_cpu_blocks."""
    num_gpu_blocks = self.scheduler_config.max_num_seqs
    num_cpu_blocks = 0
    return (num_gpu_blocks, num_cpu_blocks)