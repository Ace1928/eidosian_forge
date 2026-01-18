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
def smoothl1loss_beta_test():
    t = torch.randn(2, 3, 4, dtype=torch.double)
    return dict(fullname='SmoothL1Loss_beta', constructor=wrap_functional(lambda i: F.smooth_l1_loss(i, t.type_as(i), reduction='none', beta=0.5)), cpp_function_call='F::smooth_l1_loss(\n            i, t.to(i.options()), F::SmoothL1LossFuncOptions().reduction(torch::kNone), 0.5)', input_fn=lambda: torch.randn(2, 3, 4), cpp_var_map={'i': '_get_input()', 't': t}, reference_fn=lambda i, *_: loss_reference_fns['SmoothL1Loss'](i, t.type_as(i), reduction='none', beta=0.5), supports_forward_ad=True, pickle=False, default_dtype=torch.double)