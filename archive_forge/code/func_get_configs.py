from typing import Optional
import torch
import triton
import triton.language as tl
from xformers.triton.k_activations import (
def get_configs(block_k):
    return [triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': block_k}, num_stages=4, num_warps=2), triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': block_k}, num_stages=4, num_warps=2), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': block_k}, num_stages=3, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': block_k}, num_stages=3, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': block_k}, num_stages=3, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': block_k}, num_stages=3, num_warps=4), triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': block_k}, num_stages=3, num_warps=4)]