import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pytest
from statsmodels.datasets import elnino
from statsmodels.graphics.functional import (
@pytest.mark.slow
@pytest.mark.matplotlib
def test_hdr_alpha(close_figures):
    try:
        _, hdr = hdrboxplot(data, alpha=[0.7], seed=12345)
        extra_quant_t = np.vstack([[25.1, 26.5, 27.0, 26.4, 25.4, 24.1, 23.0, 22.0, 21.7, 22.1, 22.7, 23.8], [23.4, 24.8, 25.0, 23.9, 22.4, 21.1, 20.0, 19.3, 19.2, 19.4, 20.1, 21.3]])
        assert_almost_equal(hdr.extra_quantiles, extra_quant_t, decimal=0)
    except OSError:
        pytest.xfail('Multiprocess randomly crashes in Windows testing')