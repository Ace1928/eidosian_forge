import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.special import xlogy
from scipy.stats.contingency import (margins, expected_freq,
def test_chi2_contingency_bad_args():
    obs = np.array([[-1, 10], [1, 2]])
    assert_raises(ValueError, chi2_contingency, obs)
    obs = np.array([[0, 1], [0, 1]])
    assert_raises(ValueError, chi2_contingency, obs)
    obs = np.empty((0, 8))
    assert_raises(ValueError, chi2_contingency, obs)