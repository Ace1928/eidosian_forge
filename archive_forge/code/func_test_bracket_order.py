import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_less
from scipy import stats
import scipy.optimize._chandrupatla as _chandrupatla
from scipy.optimize._chandrupatla import _chandrupatla_minimize
from itertools import permutations
def test_bracket_order(self):
    loc = np.linspace(-1, 1, 6)[:, np.newaxis]
    brackets = np.array(list(permutations([-5, 0, 5]))).T
    res = _chandrupatla_minimize(self.f, *brackets, args=(loc,))
    assert np.all(np.isclose(res.x, loc) | (res.fun == self.f(loc, loc)))
    ref = res.x[:, 0]
    assert_allclose(*np.broadcast_arrays(res.x.T, ref), rtol=1e-15)