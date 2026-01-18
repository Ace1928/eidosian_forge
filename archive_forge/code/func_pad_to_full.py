from __future__ import absolute_import
from builtins import range, zip
from functools import partial
import autograd.numpy as np
import numpy as npo # original numpy
from autograd.extend import primitive, defvjp
from numpy.lib.stride_tricks import as_strided
from future.utils import iteritems
def pad_to_full(A, B, axes):
    A_pad = [(0, 0)] * A.ndim
    for ax_A, ax_B in zip(*axes):
        A_pad[ax_A] = (B.shape[ax_B] - 1,) * 2
    return npo.pad(A, A_pad, mode='constant')