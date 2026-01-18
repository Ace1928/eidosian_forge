from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
def quat_multiply(quat1: torch.Tensor, quat2: torch.Tensor) -> torch.Tensor:
    """Multiply a quaternion by another quaternion."""
    mat = _get_quat('_QUAT_MULTIPLY', dtype=quat1.dtype, device=quat1.device)
    reshaped_mat = mat.view((1,) * len(quat1.shape[:-1]) + mat.shape)
    return torch.sum(reshaped_mat * quat1[..., :, None, None] * quat2[..., None, :, None], dim=(-3, -2))