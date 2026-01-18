from typing import Dict, List, Tuple
import torch
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import in_wsl, is_neuron, STR_DTYPE_TO_TORCH_DTYPE
def allocate_cpu_cache(self) -> List[KVCache]:
    cpu_cache: List[KVCache] = []
    key_block_shape = self.get_key_block_shape()
    value_block_shape = self.get_value_block_shape()
    pin_memory = not in_wsl()
    if not pin_memory:
        logger.warning("Using 'pin_memory=False' as WSL is detected. This may slow down the performance.")
    for _ in range(self.num_layers):
        key_blocks = torch.empty(size=(self.num_cpu_blocks, *key_block_shape), dtype=self.dtype, pin_memory=pin_memory, device='cpu')
        value_blocks = torch.empty(size=(self.num_cpu_blocks, *value_block_shape), dtype=self.dtype, pin_memory=pin_memory, device='cpu')
        cpu_cache.append((key_blocks, value_blocks))
    return cpu_cache