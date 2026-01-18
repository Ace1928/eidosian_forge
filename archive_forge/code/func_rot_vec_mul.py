from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
def rot_vec_mul(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Applies a rotation to a vector. Written out by hand to avoid transfer to avoid AMP downcasting.

    Args:
        r: [*, 3, 3] rotation matrices
        t: [*, 3] coordinate tensors
    Returns:
        [*, 3] rotated coordinates
    """
    x, y, z = torch.unbind(t, dim=-1)
    return torch.stack([r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z, r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z, r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z], dim=-1)