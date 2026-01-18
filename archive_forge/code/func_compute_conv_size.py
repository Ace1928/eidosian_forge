from __future__ import absolute_import
from builtins import range, zip
from functools import partial
import autograd.numpy as np
import numpy as npo # original numpy
from autograd.extend import primitive, defvjp
from numpy.lib.stride_tricks import as_strided
from future.utils import iteritems
def compute_conv_size(A_size, B_size, mode):
    if mode == 'full':
        return A_size + B_size - 1
    elif mode == 'same':
        return A_size
    elif mode == 'valid':
        return abs(A_size - B_size) + 1
    else:
        raise Exception('Mode {0} not recognized'.format(mode))