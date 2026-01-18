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
def cosineembeddingloss_reference(input1, input2, target, margin=0, reduction='mean'):

    def _cos(a, b):
        cos = a.new(a.size(0))
        for i in range(0, a.size(0)):
            cos[i] = (a[i] * b[i]).sum() / (((a[i] * a[i]).sum() + 1e-12) * ((b[i] * b[i]).sum() + 1e-12)) ** 0.5
        return cos
    output = torch.where(target == 1, 1 - _cos(input1, input2), (_cos(input1, input2) - margin).clamp(min=0))
    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    return output