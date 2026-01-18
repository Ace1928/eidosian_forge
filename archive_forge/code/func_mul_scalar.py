from typing import List
import torch
from torch import Tensor
from torch._ops import ops
def mul_scalar(self, x: Tensor, y: float) -> Tensor:
    r = ops.quantized.mul_scalar(x, y)
    return r