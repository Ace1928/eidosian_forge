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
def cache_swap(self, blocks_to_swap_in: Dict[int, int], blocks_to_swap_out: Dict[int, int], blocks_to_copy: Dict[int, List[int]]) -> None:
    issued_cache_op = False
    if blocks_to_swap_in:
        self.cache_engine.swap_in(blocks_to_swap_in)
        issued_cache_op = True
    if blocks_to_swap_out:
        self.cache_engine.swap_out(blocks_to_swap_out)
        issued_cache_op = True
    if blocks_to_copy:
        self.cache_engine.copy(blocks_to_copy)
        issued_cache_op = True
    cache_events = self.cache_events if issued_cache_op else None
    if cache_events is not None:
        raise NotImplementedError('cache operations are not implemented for neuron backend.')