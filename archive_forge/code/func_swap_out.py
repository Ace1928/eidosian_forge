from typing import Dict, List, Tuple
import torch
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import in_wsl, is_neuron, STR_DTYPE_TO_TORCH_DTYPE
def swap_out(self, src_to_dst: Dict[int, int]) -> None:
    self._swap(self.gpu_cache, self.cpu_cache, src_to_dst)