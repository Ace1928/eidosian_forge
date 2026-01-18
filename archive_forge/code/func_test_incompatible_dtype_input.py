import itertools
import warnings
import numpy as np
from numpy import (arange, array, dot, zeros, identity, conjugate, transpose,
from numpy.random import random
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (solve, inv, det, lstsq, pinv, pinvh, norm,
from scipy.linalg._testutils import assert_no_overwrite
from scipy._lib._testutils import check_free_memory, IS_MUSL
from scipy.linalg.blas import HAS_ILP64
from scipy._lib.deprecation import _NoValue
def test_incompatible_dtype_input(self):
    msg = 'cannot be cast to float\\(32, 64\\)'
    for c, t in zip('SUO', ['bytes8', 'str32', 'object']):
        with assert_raises(TypeError, match=msg):
            det(np.array([['a', 'b']] * 2, dtype=c))
    with assert_raises(TypeError, match=msg):
        det(np.array([[b'a', b'b']] * 2, dtype='V'))
    with assert_raises(TypeError, match=msg):
        det(np.array([[100, 200]] * 2, dtype='datetime64[s]'))
    with assert_raises(TypeError, match=msg):
        det(np.array([[100, 200]] * 2, dtype='timedelta64[s]'))