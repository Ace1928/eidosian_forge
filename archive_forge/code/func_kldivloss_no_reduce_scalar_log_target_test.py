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
def kldivloss_no_reduce_scalar_log_target_test():
    t = torch.rand((), dtype=torch.double).log()
    return dict(fullname='KLDivLoss_no_reduce_scalar_log_target', constructor=wrap_functional(lambda i: F.kl_div(i, t.type_as(i), reduction='none', log_target=True)), cpp_function_call='F::kl_div(i, t.to(i.options()), F::KLDivFuncOptions().reduction(torch::kNone).log_target(true))', input_fn=lambda: torch.rand(()).log(), cpp_var_map={'i': '_get_input()', 't': t}, reference_fn=lambda i, *_: loss_reference_fns['KLDivLoss_log_target'](i, t.type_as(i), reduction='none'), supports_forward_ad=True, pickle=False, default_dtype=torch.double)