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
def log_abs_det_jacobian(self, x, y):
    return self.distribution.log_prob(x)