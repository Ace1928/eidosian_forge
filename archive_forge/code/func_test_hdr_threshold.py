import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pytest
from statsmodels.datasets import elnino
from statsmodels.graphics.functional import (
@pytest.mark.slow
@pytest.mark.matplotlib
def test_hdr_threshold(close_figures):
    try:
        _, hdr = hdrboxplot(data, alpha=[0.8], threshold=0.93, seed=12345)
        labels_pos = np.all(np.isin(data, hdr.outliers).reshape(data.shape), axis=1)
        outliers = labels[labels_pos]
        assert_equal([1968, 1982, 1983, 1997, 1998], outliers)
    except OSError:
        pytest.xfail('Multiprocess randomly crashes in Windows testing')