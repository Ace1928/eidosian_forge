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
def bce_with_logistic_no_reduce_test():
    t = Variable(torch.randn(15, 10).gt(0).to(torch.double))
    sigmoid = nn.Sigmoid()
    return dict(fullname='BCEWithLogitsLoss_no_reduce', constructor=wrap_functional(lambda i: F.binary_cross_entropy_with_logits(i, t.type_as(i), reduction='none')), cpp_function_call='F::binary_cross_entropy_with_logits(\n            i, t.to(i.options()), F::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kNone))', input_fn=lambda: torch.rand(15, 10).clamp_(0.028, 1 - 0.028), cpp_var_map={'i': '_get_input()', 't': t}, reference_fn=lambda i, *_: -(t * sigmoid(i).log() + (1 - t) * (1 - sigmoid(i)).log()), check_gradgrad=False, pickle=False, default_dtype=torch.double)