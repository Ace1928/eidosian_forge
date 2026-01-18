import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
from scipy.stats import variation
from scipy._lib._util import AxisError
@pytest.mark.parametrize('axis, expected', [(0, np.empty((1, 0))), (1, np.full((5, 1), fill_value=np.nan))])
def test_keepdims_size0(self, axis, expected):
    x = np.zeros((5, 0))
    y = variation(x, axis=axis, keepdims=True)
    assert_equal(y, expected)