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
def check_jacobian(self, module, input: _TensorOrTensors, jacobian_input=True):
    jacobian_parameters = bool(self._get_parameters(module)[0])
    analytical = self._analytical_jacobian(module, input, jacobian_input, jacobian_parameters)
    numerical = self._numerical_jacobian(module, input, jacobian_input, jacobian_parameters)
    analytical_t = list(_iter_tensors(analytical))
    numerical_t = list(_iter_tensors(numerical))
    differences = []
    for a, n in zip(analytical_t, numerical_t):
        if a.numel() != 0:
            differences.append(a.add(n, alpha=-1).abs().max())
    if len(differences) > 0:
        self.assertLessEqual(max(differences), PRECISION)