from typing import List, Optional
import importlib
import torch
import torch.nn as nn
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (BlockDiagonalCausalMask,
from vllm._C import ops
from vllm._C import cache_ops
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.triton_kernel.prefix_prefill import (
from vllm.utils import is_hip
def check_use_ref_attention(self) -> bool:
    if not is_hip():
        return False
    return importlib.util.find_spec('flash_attn') is None