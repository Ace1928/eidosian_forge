import numbers
import warnings
import torch
import torch.nn as nn
from torch import Tensor  # noqa: F401
from torch._jit_internal import Tuple, Optional, List, Union, Dict  # noqa: F401
from torch.nn.utils.rnn import PackedSequence
from torch.ao.nn.quantized.modules.utils import _quantize_weight
def check_forward_hidden(self, input: Tensor, hx: Tensor, hidden_label: str='') -> None:
    if input.size(0) != hx.size(0):
        raise RuntimeError(f"Input batch size {input.size(0)} doesn't match hidden{hidden_label} batch size {hx.size(0)}")
    if hx.size(1) != self.hidden_size:
        raise RuntimeError(f'hidden{hidden_label} has inconsistent hidden_size: got {hx.size(1)}, expected {self.hidden_size}')