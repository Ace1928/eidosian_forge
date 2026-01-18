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
def cross_entropy_loss_indices_target_reference(input, target, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
    log_softmax_input = torch.log_softmax(input, 1)
    nllloss = F.nll_loss(log_softmax_input, target, weight, ignore_index=ignore_index, reduction=reduction)
    if label_smoothing == 0.0:
        return nllloss
    assert 0.0 < label_smoothing <= 1.0
    input = torch.log_softmax(input, 1)
    C = input.size(1)
    if weight is not None:
        input = input * weight.view(1, C, *(1 for _ in input.shape[2:]))
    smooth_loss = -torch.sum(input, 1)
    ignore_mask = target == ignore_index
    smooth_loss.masked_fill_(ignore_mask, 0.0)
    if reduction == 'mean':
        if weight is not None:
            ret = torch.sum(smooth_loss) / weight.gather(0, target.masked_select(ignore_mask.logical_not()).flatten()).sum()
        else:
            ret = torch.mean(smooth_loss.masked_select(ignore_mask.logical_not()))
    elif reduction == 'sum':
        ret = torch.sum(smooth_loss)
    else:
        ret = smooth_loss
    return (1 - label_smoothing) * nllloss + ret * (label_smoothing / C)