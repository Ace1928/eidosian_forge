import os
import functools
import operator
from scipy._lib import _pep440
import numpy as np
from numpy.testing import assert_
import pytest
import scipy.special as sc
def get_tolerances(self, dtype):
    if not np.issubdtype(dtype, np.inexact):
        dtype = np.dtype(float)
    info = np.finfo(dtype)
    rtol, atol = (self.rtol, self.atol)
    if rtol is None:
        rtol = 5 * info.eps
    if atol is None:
        atol = 5 * info.tiny
    return (rtol, atol)