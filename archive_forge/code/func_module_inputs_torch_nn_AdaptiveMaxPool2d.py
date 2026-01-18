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
def module_inputs_torch_nn_AdaptiveMaxPool2d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    return [ModuleInput(constructor_input=FunctionInput(3), forward_input=FunctionInput(make_input((1, 3, 5, 6))), desc='single'), ModuleInput(constructor_input=FunctionInput(3), forward_input=FunctionInput(make_input((3, 5, 6))), reference_fn=no_batch_dim_reference_fn, desc='no_batch_dim'), ModuleInput(constructor_input=FunctionInput((3, 4)), forward_input=FunctionInput(make_input((1, 3, 5, 6))), desc='tuple'), ModuleInput(constructor_input=FunctionInput((3, None)), forward_input=FunctionInput(make_input((1, 3, 5, 6))), desc='tuple_none')]