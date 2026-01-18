from abc import abstractmethod
import tempfile
import unittest
from copy import deepcopy
from functools import reduce, partial, wraps
from itertools import product
from operator import mul
from math import pi
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import _reduction as _Reduction
from torch.testing._internal.common_utils import TestCase, to_gpu, freeze_rng_state, is_iterable, \
from torch.testing._internal.common_cuda import TEST_CUDA, SM90OrLater
from torch.autograd.gradcheck import _get_numerical_jacobian, _iter_tensors
from torch.autograd import Variable
from torch.types import _TensorOrTensors
import torch.backends.cudnn
from typing import Dict, Callable, Tuple, List, Sequence, Union, Any
def mseloss_no_reduce_scalar_test():
    input_size = ()
    target = torch.randn(input_size, dtype=torch.double)
    return dict(fullname='MSELoss_no_reduce_scalar', constructor=wrap_functional(lambda i: F.mse_loss(i, target.type_as(i), reduction='none')), cpp_function_call='F::mse_loss(i, target.to(i.options()), F::MSELossFuncOptions().reduction(torch::kNone))', input_size=input_size, cpp_var_map={'i': '_get_input()', 'target': target}, reference_fn=lambda i, *_: (i - target).pow(2), supports_forward_ad=True, pickle=False, default_dtype=torch.double)