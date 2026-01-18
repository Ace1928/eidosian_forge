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
def noncontiguize(self, obj):
    if isinstance(obj, list):
        return [self.noncontiguize(o) for o in obj]
    elif isinstance(obj, tuple):
        return tuple((self.noncontiguize(o) for o in obj))
    tensor = obj
    ndim = tensor.dim()
    dim = ndim
    for d in range(ndim):
        if tensor.size(d) > 1:
            dim = d + 1
            break
    noncontig = torch.stack([torch.empty_like(tensor), tensor], dim).select(dim, 1).detach()
    assert noncontig.numel() == 1 or noncontig.numel() == 0 or (not noncontig.is_contiguous())
    noncontig.requires_grad = tensor.requires_grad
    return noncontig