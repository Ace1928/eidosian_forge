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
def generate_regression_criterion_inputs(make_input):
    return [ModuleInput(constructor_input=FunctionInput(reduction=reduction), forward_input=FunctionInput(make_input((4,)), make_input(4)), reference_fn=partial(no_batch_dim_reference_fn, is_criterion=True), desc=f'no_batch_dim_{reduction}') for reduction in ['none', 'mean', 'sum']]