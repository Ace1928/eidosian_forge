import warnings
from typing import List, Optional, Tuple
import torch
from torch import _VF, Tensor  # noqa: F401
from torch.nn.utils.rnn import PackedSequence
def get_weight_bias(ihhh):
    weight_name = f'weight_{ihhh}_l{layer}{suffix}'
    bias_name = f'bias_{ihhh}_l{layer}{suffix}'
    weight = getattr(other, weight_name)
    bias = getattr(other, bias_name)
    return (weight, bias)