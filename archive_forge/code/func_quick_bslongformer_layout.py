import math
from typing import List
import numpy as np
import torch
from xformers.components.attention.sparsity_config import (
def quick_bslongformer_layout(num_heads: int, block_size: int, seq_len: int):
    config = BSLongformerSparsityConfig(num_heads=num_heads, block_size=block_size)
    return config.make_layout(seq_len)