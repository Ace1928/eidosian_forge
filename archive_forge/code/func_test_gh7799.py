import copy
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import cos, sin
from scipy.optimize import basinhopping, OptimizeResult
from scipy.optimize._basinhopping import (
def test_gh7799(self):

    def func(x):
        return (x ** 2 - 8) ** 2 + (x + 2) ** 2
    x0 = -4
    limit = 50
    con = ({'type': 'ineq', 'fun': lambda x: func(x) - limit},)
    res = basinhopping(func, x0, 30, minimizer_kwargs={'constraints': con})
    assert res.success
    assert_allclose(res.fun, limit, rtol=1e-06)