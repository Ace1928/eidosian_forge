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
def smoothl1loss_reference(input, target, reduction='mean', beta=1.0):
    abs_diff = (input - target).abs()
    ge_beta_mask = (abs_diff >= beta).type_as(abs_diff)
    lt_beta_mask = (abs_diff < beta).type_as(abs_diff)
    if beta == 0:
        output = abs_diff
    else:
        output = ge_beta_mask * (abs_diff - 0.5 * beta) + lt_beta_mask * 0.5 * abs_diff ** 2 / beta
    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    return output