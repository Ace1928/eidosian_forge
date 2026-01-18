import math
from typing import List
import numpy as np
import torch
from xformers.components.attention.sparsity_config import (
def causal_1d_pattern(attn_size: int) -> torch.Tensor:
    mask = torch.tril(torch.ones(attn_size, attn_size, dtype=torch.bool))
    return mask