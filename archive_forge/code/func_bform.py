from typing import Optional, Tuple
import torch
from torch import Tensor
def bform(X: Tensor, A: Optional[Tensor], Y: Tensor) -> Tensor:
    """Return bilinear form of matrices: :math:`X^T A Y`."""
    return matmul(transpose(X), matmul(A, Y))