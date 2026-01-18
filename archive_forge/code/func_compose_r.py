from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
def compose_r(self, r: Rotation) -> Rotation:
    """
        Compose the rotation matrices of the current Rotation object with those of another.

        Args:
            r:
                An update rotation object
        Returns:
            An updated rotation object
        """
    r1 = self.get_rot_mats()
    r2 = r.get_rot_mats()
    new_rot_mats = rot_matmul(r1, r2)
    return Rotation(rot_mats=new_rot_mats, quats=None)