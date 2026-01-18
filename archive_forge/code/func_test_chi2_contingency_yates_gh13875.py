import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.special import xlogy
from scipy.stats.contingency import (margins, expected_freq,
def test_chi2_contingency_yates_gh13875():
    observed = np.array([[1573, 3], [4, 0]])
    p = chi2_contingency(observed)[1]
    assert_allclose(p, 1, rtol=1e-12)