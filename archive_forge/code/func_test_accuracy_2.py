import pytest
import numpy as np
from scipy.optimize import quadratic_assignment, OptimizeWarning
from scipy.optimize._qap import _calc_score as _score
from numpy.testing import assert_equal, assert_, assert_warns
def test_accuracy_2(self):
    A = np.array([[0, 5, 8, 6], [5, 0, 5, 1], [8, 5, 0, 2], [6, 1, 2, 0]])
    B = np.array([[0, 1, 8, 4], [1, 0, 5, 2], [8, 5, 0, 5], [4, 2, 5, 0]])
    res = quadratic_assignment(A, B, method=self.method, options={'rng': 0, 'maximize': False})
    if self.method == 'faq':
        assert_equal(res.fun, 178)
        assert_equal(res.col_ind, np.array([1, 0, 3, 2]))
    else:
        assert_equal(res.fun, 176)
        assert_equal(res.col_ind, np.array([1, 2, 3, 0]))
    res = quadratic_assignment(A, B, method=self.method, options={'rng': 0, 'maximize': True})
    assert_equal(res.fun, 286)
    assert_equal(res.col_ind, np.array([2, 3, 0, 1]))