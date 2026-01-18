import sys
import platform
import numpy as np
from numpy.testing import (assert_, assert_allclose, assert_equal,
from pytest import raises as assert_raises
from scipy.optimize import linprog, OptimizeWarning
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse.linalg import MatrixRankWarning
from scipy.linalg import LinAlgWarning
from scipy._lib._util import VisibleDeprecationWarning
import scipy.sparse
import pytest
def generic_callback_test(self):
    last_cb = {}

    def cb(res):
        message = res.pop('message')
        complete = res.pop('complete')
        assert_(res.pop('phase') in (1, 2))
        assert_(res.pop('status') in range(4))
        assert_(isinstance(res.pop('nit'), int))
        assert_(isinstance(complete, bool))
        assert_(isinstance(message, str))
        last_cb['x'] = res['x']
        last_cb['fun'] = res['fun']
        last_cb['slack'] = res['slack']
        last_cb['con'] = res['con']
    c = np.array([-3, -2])
    A_ub = [[2, 1], [1, 1], [1, 0]]
    b_ub = [10, 8, 4]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, callback=cb, method=self.method)
    _assert_success(res, desired_fun=-18.0, desired_x=[2, 6])
    assert_allclose(last_cb['fun'], res['fun'])
    assert_allclose(last_cb['x'], res['x'])
    assert_allclose(last_cb['con'], res['con'])
    assert_allclose(last_cb['slack'], res['slack'])