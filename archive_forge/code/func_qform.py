from typing import Optional, Tuple
import torch
from torch import Tensor
def qform(A: Optional[Tensor], S: Tensor):
    """Return quadratic form :math:`S^T A S`."""
    return bform(S, A, S)