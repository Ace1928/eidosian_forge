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
def module_inputs_torch_nn_L1Loss(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    return [ModuleInput(constructor_input=FunctionInput(), forward_input=FunctionInput(make_input((2, 3, 4)), make_input((2, 3, 4))), reference_fn=lambda m, p, i, t: 1.0 / i.numel() * sum(((a - b).abs().sum() for a, b in zip(i, t)))), ModuleInput(constructor_input=FunctionInput(), forward_input=FunctionInput(make_input(()), make_input(())), reference_fn=lambda m, p, i, t: 1.0 / i.numel() * (i - t).abs().sum(), desc='scalar')] + generate_regression_criterion_inputs(make_input)