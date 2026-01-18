from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
def map_tensor_fn(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> Rigid:
    """
        Apply a Tensor -> Tensor function to underlying translation and rotation tensors, mapping over the
        translation/rotation dimensions respectively.

        Args:
            fn:
                A Tensor -> Tensor function to be mapped over the Rigid
        Returns:
            The transformed Rigid object
        """
    new_rots = self._rots.map_tensor_fn(fn)
    new_trans = torch.stack(list(map(fn, torch.unbind(self._trans, dim=-1))), dim=-1)
    return Rigid(new_rots, new_trans)