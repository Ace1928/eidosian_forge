import math
from typing import List
import numpy as np
import torch
from xformers.components.attention.sparsity_config import (
def random_pattern(attn_size: int, sparsity: float) -> torch.Tensor:
    assert 0 < sparsity < 1
    mask = torch.rand(attn_size, attn_size) > sparsity
    return mask