import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
from scipy.stats import variation
from scipy._lib._util import AxisError
@pytest.mark.parametrize('incr, expected_fill', [(0, np.inf), (1, np.nan)])
def test_keepdims_and_ddof_eq_len_plus_incr(self, incr, expected_fill):
    x = np.array([[1, 1, 2, 2], [1, 2, 3, 3]])
    y = variation(x, axis=1, ddof=x.shape[1] + incr, keepdims=True)
    assert_equal(y, np.full((2, 1), fill_value=expected_fill))