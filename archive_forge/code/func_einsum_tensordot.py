from __future__ import absolute_import
from builtins import range, zip
from functools import partial
import autograd.numpy as np
import numpy as npo # original numpy
from autograd.extend import primitive, defvjp
from numpy.lib.stride_tricks import as_strided
from future.utils import iteritems
def einsum_tensordot(A, B, axes, reverse=False):
    A_axnums = list(range(A.ndim))
    B_axnums = list(range(A.ndim, A.ndim + B.ndim))
    sum_axnum = A.ndim + B.ndim
    for i_sum, (i_A, i_B) in enumerate(zip(*axes)):
        A_axnums[i_A] = sum_axnum + i_sum
        B_axnums[i_B] = sum_axnum + i_sum
    return npo.einsum(A, A_axnums, B, B_axnums)