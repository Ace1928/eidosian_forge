import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.special import xlogy
from scipy.stats.contingency import (margins, expected_freq,
def test_chi2_contingency_R():
    obs = np.array([[[12, 34, 23], [35, 31, 11], [12, 32, 9], [12, 12, 14]], [[4, 47, 11], [34, 10, 18], [18, 13, 19], [9, 33, 25]]])
    chi2, p, dof, expected = chi2_contingency(obs)
    assert_approx_equal(chi2, 102.17, significant=5)
    assert_approx_equal(p, 3.514e-14, significant=4)
    assert_equal(dof, 17)
    obs = np.array([[[[12, 17], [11, 16]], [[11, 12], [15, 16]]], [[[23, 15], [30, 22]], [[14, 17], [15, 16]]]])
    chi2, p, dof, expected = chi2_contingency(obs)
    assert_approx_equal(chi2, 8.758, significant=4)
    assert_approx_equal(p, 0.6442, significant=4)
    assert_equal(dof, 11)