from functools import reduce  # Required in Python 3
import operator
from typing import Optional
import warnings
import torch
from bitsandbytes.autograd._functions import GlobalOutlierPooler, MatmulLtState
import bitsandbytes.functional as F
def switchback_bnb(A: torch.Tensor, B: torch.Tensor, out: Optional[torch.Tensor]=None, state: Optional[MatmulLtState]=None, threshold=0.0, bias=None):
    state = state or MatmulLtState()
    if threshold > 0.0:
        state.threshold = threshold
    return SwitchBackBnb.apply(A, B, out, bias, state)