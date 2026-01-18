import torch
import unittest
from copy import deepcopy
from enum import Enum
from functools import wraps, partial
from itertools import chain, product
import itertools
import math
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import TEST_CUDNN
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_device_type import (
from torch.testing._internal.common_methods_invocations import DecorateInfo
from torch.testing._internal.common_nn import nllloss_reference, get_reduction
from torch.testing._internal.common_utils import (
from types import ModuleType
from typing import List, Tuple, Type, Set, Dict
def no_batch_dim_reference_lstm(m, p, *args, **kwargs):
    """Reference function for LSTM supporting no batch dimensions.

    Unbatched inputs are unsqueezed to form a
    single batch input before passing them to the module.
    The output is squeezed to compare with the
    output of unbatched input to the module.
    """
    if len(args) == 1:
        inp, = args
        h = None
    elif len(args) == 2:
        inp, h = args
        h = (h[0].unsqueeze(1), h[1].unsqueeze(1))
    batch_dim = 0 if kwargs['batch_first'] else 1
    kwargs.pop('batch_first')
    inp = inp.unsqueeze(batch_dim)
    single_batch_input_args = (inp, h)
    with freeze_rng_state():
        output = m(*single_batch_input_args, **kwargs)
        return (output[0].squeeze(batch_dim), (output[1][0].squeeze(1), output[1][1].squeeze(1)))