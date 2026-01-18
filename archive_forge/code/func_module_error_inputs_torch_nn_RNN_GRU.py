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
def module_error_inputs_torch_nn_RNN_GRU(module_info, device, dtype, requires_grad, training, **kwargs):
    samples = [ErrorModuleInput(ModuleInput(constructor_input=FunctionInput(10, 0, 1)), error_on=ModuleErrorEnum.CONSTRUCTION_ERROR, error_type=ValueError, error_regex='hidden_size must be greater than zero'), ErrorModuleInput(ModuleInput(constructor_input=FunctionInput(10, 10, 0)), error_on=ModuleErrorEnum.CONSTRUCTION_ERROR, error_type=ValueError, error_regex='num_layers must be greater than zero')]
    return samples