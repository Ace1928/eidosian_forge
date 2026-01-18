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
def module_inputs_torch_nn_BatchNorm3d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    return [ModuleInput(constructor_input=FunctionInput(3), forward_input=FunctionInput(make_input((2, 3, 4, 4, 4)))), ModuleInput(constructor_input=FunctionInput(3, 0.001, None), forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))), desc='3d_simple_average'), ModuleInput(constructor_input=FunctionInput(3, 0.001, 0.7), forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))), desc='momentum'), ModuleInput(constructor_input=FunctionInput(3, 0.001, 0.7, False), forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))), desc='not_affine'), ModuleInput(constructor_input=FunctionInput(3, 0.001, 0.7, True, False), forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))), desc='not_tracking_stats'), ModuleInput(constructor_input=FunctionInput(5, 0.001, 0.3, False), forward_input=FunctionInput(make_input((0, 5, 2, 2, 2))), desc='zero_batch')]