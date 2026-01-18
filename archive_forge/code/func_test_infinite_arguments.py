import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import scipy.special as sc
from scipy.special._testutils import FuncData
@pytest.mark.parametrize('a, x, desired', [(np.inf, 1, 1), (np.inf, 0, 1), (np.inf, np.inf, np.nan), (1, np.inf, 0)])
def test_infinite_arguments(self, a, x, desired):
    result = sc.gammaincc(a, x)
    if np.isnan(desired):
        assert np.isnan(result)
    else:
        assert result == desired