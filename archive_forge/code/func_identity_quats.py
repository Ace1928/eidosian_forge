from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
@lru_cache(maxsize=None)
def identity_quats(batch_dims: Tuple[int, ...], dtype: Optional[torch.dtype]=None, device: Optional[torch.device]=None, requires_grad: bool=True) -> torch.Tensor:
    quat = torch.zeros((*batch_dims, 4), dtype=dtype, device=device, requires_grad=requires_grad)
    with torch.no_grad():
        quat[..., 0] = 1
    return quat