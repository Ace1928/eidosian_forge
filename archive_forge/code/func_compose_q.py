from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
def compose_q(self, r: Rotation, normalize_quats: bool=True) -> Rotation:
    """
        Compose the quaternions of the current Rotation object with those of another.

        Depending on whether either Rotation was initialized with quaternions, this function may call
        torch.linalg.eigh.

        Args:
            r:
                An update rotation object
        Returns:
            An updated rotation object
        """
    q1 = self.get_quats()
    q2 = r.get_quats()
    new_quats = quat_multiply(q1, q2)
    return Rotation(rot_mats=None, quats=new_quats, normalize_quats=normalize_quats)