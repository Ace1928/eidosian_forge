from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
def get_rots(self) -> Rotation:
    """
        Getter for the rotation.

        Returns:
            The rotation object
        """
    return self._rots