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
def module_inputs_torch_nn_ConvNd(module_info, device, dtype, requires_grad, training, **kwargs):
    N = kwargs['N']
    lazy = kwargs.get('lazy', False)
    transposed = kwargs.get('transposed', False)
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    conv_kwargs_list = [{}] if transposed else [{}, {'padding': 'same'}]
    kernel_size, C_in, C_out = (3, 4, 5)
    input_no_batch_shape = (C_in,) + tuple((i + 3 for i in range(N)))
    input_batch_shape = (2,) + input_no_batch_shape
    return [ModuleInput(constructor_input=FunctionInput(C_out, kernel_size, **conv_kwargs) if lazy else FunctionInput(C_in, C_out, kernel_size, **conv_kwargs), forward_input=FunctionInput(make_input(input_batch_shape if with_batch else input_no_batch_shape)), desc='' if with_batch else 'no_batch_dim', reference_fn=None if with_batch else no_batch_dim_reference_fn) for with_batch, conv_kwargs in itertools.product([True, False], conv_kwargs_list)]