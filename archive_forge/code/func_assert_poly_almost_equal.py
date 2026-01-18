import operator as op
from numbers import Number
import pytest
import numpy as np
from numpy.polynomial import (
from numpy.testing import (
from numpy.polynomial.polyutils import RankWarning
def assert_poly_almost_equal(p1, p2, msg=''):
    try:
        assert_(np.all(p1.domain == p2.domain))
        assert_(np.all(p1.window == p2.window))
        assert_almost_equal(p1.coef, p2.coef)
    except AssertionError:
        msg = f'Result: {p1}\nTarget: {p2}'
        raise AssertionError(msg)