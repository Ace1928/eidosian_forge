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
def module_inputs_torch_nn_Linear(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    module_inputs = [ModuleInput(constructor_input=FunctionInput(10, 8), forward_input=FunctionInput(input=make_input((4, 10))), reference_fn=lambda m, p, input: torch.mm(input, p[0].t()) + p[1].view(1, -1).expand(4, 8)), ModuleInput(constructor_input=FunctionInput(10, 8, bias=False), forward_input=FunctionInput(make_input((4, 10))), desc='no_bias', reference_fn=lambda m, p, i: torch.mm(i, p[0].t())), ModuleInput(constructor_input=FunctionInput(3, 5), forward_input=FunctionInput(make_input(3)), desc='no_batch_dim', reference_fn=lambda m, p, i: torch.mm(i.view(1, -1), p[0].t()).view(-1) + p[1])]
    return module_inputs