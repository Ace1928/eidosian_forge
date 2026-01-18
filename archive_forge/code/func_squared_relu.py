import math
from enum import Enum
from typing import Optional
import triton
import triton.language as tl
@triton.jit
def squared_relu(x):
    """
    Squared ReLU activation, as proposed in the Primer_ paper.

    .. _Primer: https://arxiv.org/abs/2109.08668
    """
    x_ = relu(x)
    return (x_ * x_).to(x.dtype)