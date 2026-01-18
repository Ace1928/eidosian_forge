import numpy as np
from numpy.testing import assert_equal, assert_allclose
import scipy.special as sc
def test_log_ndtr_values_gt31(self):
    x = np.array([31.6, 32.8, 34.9, 37.1])
    expected = [-1.846036234858162e-219, -2.9440539964066835e-236, -3.71721649450857e-267, -1.4047119663106221e-301]
    y = sc.log_ndtr(x)
    assert_allclose(y, expected, rtol=3e-13)