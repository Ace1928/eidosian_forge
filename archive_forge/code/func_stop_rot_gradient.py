from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
def stop_rot_gradient(self) -> Rigid:
    """
        Detaches the underlying rotation object

        Returns:
            A transformation object with detached rotations
        """
    return self.apply_rot_fn(lambda r: r.detach())