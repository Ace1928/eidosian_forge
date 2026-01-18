from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
def rot_to_quat(rot: torch.Tensor) -> torch.Tensor:
    if rot.shape[-2:] != (3, 3):
        raise ValueError('Input rotation is incorrectly shaped')
    [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = [[rot[..., i, j] for j in range(3)] for i in range(3)]
    k = [[xx + yy + zz, zy - yz, xz - zx, yx - xy], [zy - yz, xx - yy - zz, xy + yx, xz + zx], [xz - zx, xy + yx, yy - xx - zz, yz + zy], [yx - xy, xz + zx, yz + zy, zz - xx - yy]]
    _, vectors = torch.linalg.eigh(1.0 / 3.0 * torch.stack([torch.stack(t, dim=-1) for t in k], dim=-2))
    return vectors[..., -1]