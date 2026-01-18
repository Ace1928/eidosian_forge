import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.imports import _NLTK_AVAILABLE
def lcs_ind(pred_tokens: Sequence[str], target_tokens: Sequence[str]) -> Sequence[int]:
    """Return one of the longest of longest common subsequence via backtracked lcs table."""
    lcs_table: Sequence[Sequence[int]] = _lcs(pred_tokens, target_tokens, return_full_table=True)
    return _backtracked_lcs(lcs_table, pred_tokens, target_tokens)