from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
def compose_q_update_vec(self, q_update_vec: torch.Tensor) -> Rigid:
    """
        Composes the transformation with a quaternion update vector of shape [*, 6], where the final 6 columns
        represent the x, y, and z values of a quaternion of form (1, x, y, z) followed by a 3D translation.

        Args:
            q_vec: The quaternion update vector.
        Returns:
            The composed transformation.
        """
    q_vec, t_vec = (q_update_vec[..., :3], q_update_vec[..., 3:])
    new_rots = self._rots.compose_q_update_vec(q_vec)
    trans_update = self._rots.apply(t_vec)
    new_translation = self._trans + trans_update
    return Rigid(new_rots, new_translation)