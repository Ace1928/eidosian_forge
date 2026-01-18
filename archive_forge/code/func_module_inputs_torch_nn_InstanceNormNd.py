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
def module_inputs_torch_nn_InstanceNormNd(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    lazy = kwargs.get('lazy', False)
    N = kwargs['N']
    num_features, eps, momentum, affine, track_running_stats = (3, 0.001, 0.3, False, True)
    input_no_batch_shape_dict = {1: (3, 15), 2: (3, 6, 6), 3: (3, 4, 4, 4)}
    input_no_batch_shape = input_no_batch_shape_dict[N]
    input_batch_shape = (4,) + input_no_batch_shape
    return [ModuleInput(constructor_input=FunctionInput(eps, momentum) if lazy else FunctionInput(num_features, eps, momentum), forward_input=FunctionInput(make_input(input_batch_shape))), ModuleInput(constructor_input=FunctionInput(eps, momentum, affine, track_running_stats) if lazy else FunctionInput(num_features, eps, momentum, affine, track_running_stats), forward_input=FunctionInput(make_input(input_batch_shape)), desc='tracking_stats'), ModuleInput(constructor_input=FunctionInput(eps, momentum) if lazy else FunctionInput(num_features, eps, momentum), forward_input=FunctionInput(make_input(input_no_batch_shape)), reference_fn=no_batch_dim_reference_fn, desc='tracking_stats_no_batch_dim'), ModuleInput(constructor_input=FunctionInput(eps, momentum, affine, track_running_stats) if lazy else FunctionInput(num_features, eps, momentum, affine, track_running_stats), forward_input=FunctionInput(make_input(input_no_batch_shape)), reference_fn=no_batch_dim_reference_fn, desc='no_batch_dim')]