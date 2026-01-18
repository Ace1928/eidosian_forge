import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
from scipy.stats import variation
from scipy._lib._util import AxisError
@pytest.mark.parametrize('nan_policy', ['propagate', 'omit'])
def test_combined_edge_cases(self, nan_policy):
    x = np.array([[0, 10, np.nan, 1], [0, -5, np.nan, 2], [0, -5, np.nan, 3]])
    y = variation(x, axis=0, nan_policy=nan_policy)
    assert_allclose(y, [np.nan, np.inf, np.nan, np.sqrt(2 / 3) / 2])