from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
def invert_apply(self, pts: torch.Tensor) -> torch.Tensor:
    """
        Applies the inverse of the transformation to a coordinate tensor.

        Args:
            pts: A [*, 3] coordinate tensor
        Returns:
            The transformed points.
        """
    pts = pts - self._trans
    return self._rots.invert_apply(pts)