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
def module_inputs_torch_nn_Mish(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    return [ModuleInput(constructor_input=FunctionInput(), forward_input=FunctionInput(make_input((5, 6, 7))), reference_fn=lambda m, p, i: i * torch.tanh(F.softplus(i))), ModuleInput(constructor_input=FunctionInput(), forward_input=FunctionInput(make_input(())), reference_fn=lambda m, p, i: i * torch.tanh(F.softplus(i)), desc='scalar'), ModuleInput(constructor_input=FunctionInput(), forward_input=FunctionInput(make_input(4)), reference_fn=no_batch_dim_reference_fn, desc='no_batch_dim')]