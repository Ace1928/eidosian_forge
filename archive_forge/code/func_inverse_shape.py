import functools
import math
import numbers
import operator
import weakref
from typing import List
import torch
import torch.nn.functional as F
from torch.distributions import constraints
from torch.distributions.utils import (
from torch.nn.functional import pad, softplus
def inverse_shape(self, shape):
    if len(shape) < 1:
        raise ValueError('Too few dimensions on input')
    return shape[:-1] + (shape[-1] - 1,)