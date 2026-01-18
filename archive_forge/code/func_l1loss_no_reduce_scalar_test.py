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
def l1loss_no_reduce_scalar_test():
    t = torch.randn((), dtype=torch.double)
    return dict(fullname='L1Loss_no_reduce_scalar', constructor=wrap_functional(lambda i: F.l1_loss(i, t.type_as(i), reduction='none')), cpp_function_call='F::l1_loss(i, t.to(i.options()), F::L1LossFuncOptions().reduction(torch::kNone))', input_fn=lambda: torch.randn(()), cpp_var_map={'i': '_get_input()', 't': t}, reference_fn=lambda i, *_: (i - t.type_as(i)).abs(), supports_forward_ad=True, pickle=False, default_dtype=torch.double)