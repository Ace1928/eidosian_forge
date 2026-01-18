from typing import Dict, List, Tuple
import torch
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import in_wsl, is_neuron, STR_DTYPE_TO_TORCH_DTYPE
@staticmethod
def get_cache_block_size(block_size: int, cache_dtype: str, model_config: ModelConfig, parallel_config: ParallelConfig) -> int:
    head_size = model_config.get_head_size()
    num_heads = model_config.get_num_kv_heads(parallel_config)
    num_layers = model_config.get_num_layers(parallel_config)
    key_cache_block = block_size * num_heads * head_size
    value_cache_block = key_cache_block
    total = num_layers * (key_cache_block + value_cache_block)
    if cache_dtype == 'auto':
        dtype = model_config.dtype
    else:
        dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
    dtype_size = _get_dtype_size(dtype)
    return dtype_size * total