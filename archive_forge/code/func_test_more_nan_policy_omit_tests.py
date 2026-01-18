import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
from scipy.stats import variation
from scipy._lib._util import AxisError
@pytest.mark.parametrize('ddof, expected', [(0, [np.sqrt(1 / 6), np.sqrt(5 / 8), np.inf, 0, np.nan, 0.0, np.nan]), (1, [0.5, np.sqrt(5 / 6), np.inf, 0, np.nan, 0, np.nan]), (2, [np.sqrt(0.5), np.sqrt(5 / 4), np.inf, np.nan, np.nan, 0, np.nan])])
def test_more_nan_policy_omit_tests(self, ddof, expected):
    nan = np.nan
    x = np.array([[1.0, 2.0, nan, 3.0], [0.0, 4.0, 3.0, 1.0], [nan, -0.5, 0.5, nan], [nan, 9.0, 9.0, nan], [nan, nan, nan, nan], [3.0, 3.0, 3.0, 3.0], [0.0, 0.0, 0.0, 0.0]])
    v = variation(x, axis=1, ddof=ddof, nan_policy='omit')
    assert_allclose(v, expected)