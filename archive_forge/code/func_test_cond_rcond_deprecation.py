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
@pytest.mark.parametrize('cond', [1, None, _NoValue])
@pytest.mark.parametrize('rcond', [1, None, _NoValue])
def test_cond_rcond_deprecation(self, cond, rcond):
    if cond is _NoValue and rcond is _NoValue:
        pinv(np.ones((2, 2)), cond=cond, rcond=rcond)
    else:
        with pytest.deprecated_call(match='"cond" and "rcond"'):
            pinv(np.ones((2, 2)), cond=cond, rcond=rcond)