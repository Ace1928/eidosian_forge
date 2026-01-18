import numbers
import warnings
import torch
import torch.nn as nn
from torch import Tensor  # noqa: F401
from torch._jit_internal import Tuple, Optional, List, Union, Dict  # noqa: F401
from torch.nn.utils.rnn import PackedSequence
from torch.ao.nn.quantized.modules.utils import _quantize_weight
@torch.jit.export
def forward_packed(self, input: PackedSequence, hx: Optional[Tensor]=None) -> Tuple[PackedSequence, Tensor]:
    input_, batch_sizes, sorted_indices, unsorted_indices = input
    max_batch_size = int(batch_sizes[0])
    output_, hidden = self.forward_impl(input_, hx, batch_sizes, max_batch_size, sorted_indices)
    output = PackedSequence(output_, batch_sizes, sorted_indices, unsorted_indices)
    return (output, self.permute_hidden(hidden, unsorted_indices))