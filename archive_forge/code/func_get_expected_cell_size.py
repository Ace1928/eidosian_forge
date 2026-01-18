import torch
import torch.nn as nn
from torch import Tensor
from .utils import _quantize_and_dequantize_weight
from .utils import _quantize_weight
from typing import Optional, Dict, Any, Tuple
from torch import _VF
from torch.nn.utils.rnn import PackedSequence
def get_expected_cell_size(self, input: Tensor, batch_sizes: Optional[Tensor]) -> Tuple[int, int, int]:
    if batch_sizes is not None:
        mini_batch = int(batch_sizes[0])
    else:
        mini_batch = input.size(0) if self.batch_first else input.size(1)
    num_directions = 2 if self.bidirectional else 1
    expected_hidden_size = (self.num_layers * num_directions, mini_batch, self.hidden_size)
    return expected_hidden_size