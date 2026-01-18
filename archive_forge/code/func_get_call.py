from torch.jit.annotations import BroadcastingList2, BroadcastingList3  # noqa: F401
import torch.nn.functional as F
import torch
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
from torch.testing._internal.common_nn import module_tests, new_module_tests
from torch.testing._internal.common_utils import is_iterable_of_tensors
import collections
from copy import deepcopy
from typing import Any, Dict, List, Union
import math  # noqa: F401
from torch import inf
def get_call(method_name, func_type, args, kwargs):
    kwargs_str = ', '.join([k + '=' + value_to_literal(v) for k, v in kwargs.items()])
    self_arg = args[0]
    if func_type == 'method':
        args = args[1:]
    argument_str = ', '.join(args)
    argument_str += ', ' if len(args) and len(kwargs) else ''
    argument_str += kwargs_str
    if func_type == 'functional' or func_type == 'function':
        call = f'torch.{method_name}({argument_str})'
    elif func_type == 'method':
        call = f'{self_arg}.{method_name}({argument_str})'
    elif func_type == 'nn_functional':
        call = f'torch.nn.functional.{method_name}({argument_str})'
    else:
        raise TypeError('Unsupported function type')
    return call