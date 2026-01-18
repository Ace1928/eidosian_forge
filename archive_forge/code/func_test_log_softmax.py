import numpy as np
from numpy.testing import assert_allclose
import pytest
import scipy.special as sc
@pytest.mark.parametrize('x, expected', [(np.array([1000, 1]), np.array([0, -999])), (np.arange(4), np.array([-3.4401896985611953, -2.4401896985611953, -1.4401896985611953, -0.44018969856119533]))])
def test_log_softmax(x, expected):
    assert_allclose(sc.log_softmax(x), expected, rtol=1e-13)