from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
@property
def requires_grad(self) -> bool:
    """
        Returns the requires_grad property of the underlying rotation

        Returns:
            The requires_grad property of the underlying tensor
        """
    if self._rot_mats is not None:
        return self._rot_mats.requires_grad
    elif self._quats is not None:
        return self._quats.requires_grad
    else:
        raise ValueError('Both rotations are None')