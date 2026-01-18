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
def get_nn_mod_test_name(**kwargs):
    if 'fullname' in kwargs:
        test_name = kwargs['fullname']
    else:
        test_name = get_nn_module_name_from_kwargs(**kwargs)
        if 'desc' in kwargs:
            test_name = f'{test_name}_{kwargs['desc']}'
    return f'test_nn_{test_name}'