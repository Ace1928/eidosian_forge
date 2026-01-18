from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
def scale_translation(self, trans_scale_factor: float) -> Rigid:
    """
        Scales the translation by a constant factor.

        Args:
            trans_scale_factor:
                The constant factor
        Returns:
            A transformation object with a scaled translation.
        """
    return self.apply_trans_fn(lambda t: t * trans_scale_factor)