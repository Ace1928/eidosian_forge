from typing import Dict, List, Tuple
import torch
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import in_wsl, is_neuron, STR_DTYPE_TO_TORCH_DTYPE
def swap_in(self, src_to_dst: Dict[int, int]) -> None:
    self._swap(self.cpu_cache, self.gpu_cache, src_to_dst)