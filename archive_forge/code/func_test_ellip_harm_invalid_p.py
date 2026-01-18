import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose,
from scipy.special._testutils import assert_func_equal
from scipy.special import ellip_harm, ellip_harm_2, ellip_normal
from scipy.integrate import IntegrationWarning
from numpy import sqrt, pi
def test_ellip_harm_invalid_p():
    n = 4
    p = 2 * n + 2
    result = ellip_harm(0.5, 2.0, n, p, 0.2)
    assert np.isnan(result)