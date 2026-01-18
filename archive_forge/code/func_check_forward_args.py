import numbers
import warnings
import torch
import torch.nn as nn
from torch import Tensor  # noqa: F401
from torch._jit_internal import Tuple, Optional, List, Union, Dict  # noqa: F401
from torch.nn.utils.rnn import PackedSequence
from torch.ao.nn.quantized.modules.utils import _quantize_weight
def check_forward_args(self, input: Tensor, hidden: Tensor, batch_sizes: Optional[Tensor]) -> None:
    self.check_input(input, batch_sizes)
    expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)
    self.check_hidden_size(hidden, expected_hidden_size, 'Expected hidden size {}, got {}')