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
def cross_entropy_loss_prob_target_reference(input, target, weight=None, reduction='mean', label_smoothing=0.0):
    assert input.dim() >= 2
    input = torch.log_softmax(input, 1)
    C = input.size(1)
    if weight is None:
        weight = torch.ones(C).type_as(input)
    weight = weight.view(1, C, *(1 for _ in input.shape[2:]))
    if label_smoothing > 0.0:
        assert label_smoothing <= 1.0
        target = target * (1 - label_smoothing) + label_smoothing / C
    output = -(input * target * weight).sum(dim=1)
    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    return output