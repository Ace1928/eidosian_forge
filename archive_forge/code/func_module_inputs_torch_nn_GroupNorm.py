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
def module_inputs_torch_nn_GroupNorm(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    return [ModuleInput(constructor_input=FunctionInput(3, 6, 0.001), forward_input=FunctionInput(make_input((4, 6, 5))), desc='1d_affine'), ModuleInput(constructor_input=FunctionInput(3, 12, 0.001), forward_input=FunctionInput(make_input((4, 12))), desc='1d_affine_GN'), ModuleInput(constructor_input=FunctionInput(1, 6, 0.001), forward_input=FunctionInput(make_input((150, 6))), desc='1d_affine_large_batch'), ModuleInput(constructor_input=FunctionInput(5, 5, 0.001, False), forward_input=FunctionInput(make_input((4, 5, 5))), desc='1d_no_affine_IN'), ModuleInput(constructor_input=FunctionInput(1, 10, 0.001, False), forward_input=FunctionInput(make_input((4, 10))), desc='1d_no_affine_LN'), ModuleInput(constructor_input=FunctionInput(3, 6, 0.001), forward_input=FunctionInput(make_input((4, 6, 2, 3))), desc='2d_affine'), ModuleInput(constructor_input=FunctionInput(3, 6, 0.001), forward_input=FunctionInput(make_input((4, 6, 28, 28))), desc='2d_affine_large_feature'), ModuleInput(constructor_input=FunctionInput(3, 51, 1e-05, False), forward_input=FunctionInput(make_input((2, 51, 28, 28))), desc='2d_no_affine_large_feature'), ModuleInput(constructor_input=FunctionInput(3, 3, 0.001, False), forward_input=FunctionInput(make_input((4, 3, 2, 3))), desc='2d_no_affine_IN'), ModuleInput(constructor_input=FunctionInput(1, 3, 0.001, False), forward_input=FunctionInput(make_input((4, 3, 2, 3))), desc='2d_no_affine_LN')]