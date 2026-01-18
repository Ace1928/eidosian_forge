from itertools import permutations
from typing import Any, Callable, Tuple
import numpy as np
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import _SCIPY_AVAILABLE
def pit_permutate(preds: Tensor, perm: Tensor) -> Tensor:
    """Permutate estimate according to perm.

    Args:
        preds: the estimates you want to permutate, shape [batch, spk, ...]
        perm: the permutation returned from permutation_invariant_training, shape [batch, spk]

    Returns:
        Tensor: the permutated version of estimate

    """
    return torch.stack([torch.index_select(pred, 0, p) for pred, p in zip(preds, perm)])