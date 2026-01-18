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
def module_inputs_torch_nn_RNN_GRU_Cell(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    samples = [ModuleInput(constructor_input=FunctionInput(5, 10), forward_input=FunctionInput(make_input(5), make_input(10)), reference_fn=no_batch_dim_reference_fn), ModuleInput(constructor_input=FunctionInput(5, 10, bias=True), forward_input=FunctionInput(make_input(5), make_input(10)), reference_fn=no_batch_dim_reference_fn)]
    is_rnn = kwargs.get('is_rnn', False)
    if is_rnn:
        samples.append(ModuleInput(constructor_input=FunctionInput(5, 10, bias=True, nonlinearity='relu'), forward_input=FunctionInput(make_input(5), make_input(10)), reference_fn=no_batch_dim_reference_fn))
    return samples