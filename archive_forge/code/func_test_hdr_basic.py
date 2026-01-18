import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pytest
from statsmodels.datasets import elnino
from statsmodels.graphics.functional import (
@pytest.mark.matplotlib
def test_hdr_basic(close_figures):
    try:
        _, hdr = hdrboxplot(data, labels=labels, seed=12345)
        assert len(hdr.extra_quantiles) == 0
        median_t = [24.247, 25.625, 25.964, 24.999, 23.648, 22.302, 21.231, 20.366, 20.168, 20.434, 21.111, 22.299]
        assert_almost_equal(hdr.median, median_t, decimal=2)
        quant = np.vstack([hdr.outliers, hdr.hdr_90, hdr.hdr_50])
        quant_t = np.vstack([[24.36, 25.42, 25.4, 24.96, 24.21, 23.35, 22.5, 21.89, 22.04, 22.88, 24.57, 25.89], [27.25, 28.23, 28.85, 28.82, 28.37, 27.43, 25.73, 23.88, 22.26, 22.22, 22.21, 23.19], [23.7, 26.08, 27.17, 26.74, 26.77, 26.15, 25.59, 24.95, 24.69, 24.64, 25.85, 27.08], [28.12, 28.82, 29.24, 28.45, 27.36, 25.19, 23.61, 22.27, 21.31, 21.37, 21.6, 22.81], [25.48, 26.99, 27.51, 27.04, 26.23, 24.94, 23.69, 22.72, 22.26, 22.64, 23.33, 24.44], [23.11, 24.5, 24.66, 23.44, 21.74, 20.58, 19.68, 18.84, 18.76, 18.99, 19.66, 20.86], [24.84, 26.23, 26.67, 25.93, 24.87, 23.57, 22.46, 21.45, 21.26, 21.57, 22.14, 23.41], [23.62, 25.1, 25.34, 24.22, 22.74, 21.52, 20.4, 19.56, 19.63, 19.67, 20.37, 21.76]])
        assert_almost_equal(quant, quant_t, decimal=0)
        labels_pos = np.all(np.isin(data, hdr.outliers).reshape(data.shape), axis=1)
        outliers = labels[labels_pos]
        assert_equal([1982, 1983, 1997, 1998], outliers)
        assert_equal(labels[hdr.outliers_idx], outliers)
    except OSError:
        pytest.xfail('Multiprocess randomly crashes in Windows testing')