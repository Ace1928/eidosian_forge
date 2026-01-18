from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
def quat_to_rot(quat: torch.Tensor) -> torch.Tensor:
    """
    Converts a quaternion to a rotation matrix.

    Args:
        quat: [*, 4] quaternions
    Returns:
        [*, 3, 3] rotation matrices
    """
    quat = quat[..., None] * quat[..., None, :]
    mat = _get_quat('_QTR_MAT', dtype=quat.dtype, device=quat.device)
    shaped_qtr_mat = mat.view((1,) * len(quat.shape[:-2]) + mat.shape)
    quat = quat[..., None, None] * shaped_qtr_mat
    return torch.sum(quat, dim=(-3, -4))