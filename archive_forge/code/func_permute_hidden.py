import numbers
import warnings
import torch
import torch.nn as nn
from torch import Tensor  # noqa: F401
from torch._jit_internal import Tuple, Optional, List, Union, Dict  # noqa: F401
from torch.nn.utils.rnn import PackedSequence
from torch.ao.nn.quantized.modules.utils import _quantize_weight
def permute_hidden(self, hx: Tensor, permutation: Optional[Tensor]) -> Tensor:
    if permutation is None:
        return hx
    return _apply_permutation(hx, permutation)