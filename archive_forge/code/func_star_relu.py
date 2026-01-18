import math
from typing import Optional
import triton
import triton.language as tl
from xformers.components import Activation
@triton.jit
def star_relu(x):
    """
    Star ReLU activation, as proposed in the "MetaFormer Baselines for Vision"_ paper.

    .. _ "MetaFormer Baselines for Vision": https://arxiv.org/pdf/2210.13452.pdf
    """
    x_sq = x * x
    return 0.8944 * tl.where(x > 0.0, x_sq, 0.0) - 0.4472