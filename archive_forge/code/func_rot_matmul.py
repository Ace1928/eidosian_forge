from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
def rot_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication of two rotation matrix tensors. Written out by hand to avoid AMP downcasting.

    Args:
        a: [*, 3, 3] left multiplicand
        b: [*, 3, 3] right multiplicand
    Returns:
        The product ab
    """

    def row_mul(i: int) -> torch.Tensor:
        return torch.stack([a[..., i, 0] * b[..., 0, 0] + a[..., i, 1] * b[..., 1, 0] + a[..., i, 2] * b[..., 2, 0], a[..., i, 0] * b[..., 0, 1] + a[..., i, 1] * b[..., 1, 1] + a[..., i, 2] * b[..., 2, 1], a[..., i, 0] * b[..., 0, 2] + a[..., i, 1] * b[..., 1, 2] + a[..., i, 2] * b[..., 2, 2]], dim=-1)
    return torch.stack([row_mul(0), row_mul(1), row_mul(2)], dim=-2)