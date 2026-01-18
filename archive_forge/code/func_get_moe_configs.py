import functools
import json
import os
from typing import Any, Dict, Optional, Tuple
import torch
import triton
import triton.language as tl
from vllm._C import ops
from vllm.logger import init_logger
from vllm.utils import is_hip
@functools.lru_cache
def get_moe_configs(E: int, N: int) -> Optional[Dict[int, Any]]:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of batch sizes
    to configurations of the fused_moe kernel. To evaluate the kernel on a given batch
    size bs, the closest batch size in the grid should be picked and the associated
    configuration chosen to invoke the kernel.
    """
    device_name = torch.cuda.get_device_name().replace(' ', '_')
    config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs', f'E={E},N={N},device_name={device_name}.json')
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            logger.info(f'Using configuration from {config_file_path} for MoE layer.')
            return {int(key): val for key, val in json.load(f).items()}
    return None