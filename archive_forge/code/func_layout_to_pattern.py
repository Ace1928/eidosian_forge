import math
from typing import List
import numpy as np
import torch
from xformers.components.attention.sparsity_config import (
def layout_to_pattern(layout: torch.Tensor, block_size: int):
    """
    create a pattern of shape [heads, seq, seq] out of a blocksparse
    layout of shape [heads, seq/block_size, seq/block_size]
    """
    return torch.kron(layout, torch.ones(block_size, block_size))