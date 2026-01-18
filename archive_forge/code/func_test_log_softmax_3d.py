import numpy as np
from numpy.testing import assert_allclose
import pytest
import scipy.special as sc
def test_log_softmax_3d(log_softmax_2d_x, log_softmax_2d_expected):
    x_3d = log_softmax_2d_x.reshape(2, 2, 2)
    expected_3d = log_softmax_2d_expected.reshape(2, 2, 2)
    assert_allclose(sc.log_softmax(x_3d, axis=(1, 2)), expected_3d, rtol=1e-13)