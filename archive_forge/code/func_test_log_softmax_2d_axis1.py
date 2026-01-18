import numpy as np
from numpy.testing import assert_allclose
import pytest
import scipy.special as sc
def test_log_softmax_2d_axis1(log_softmax_2d_x, log_softmax_2d_expected):
    x = log_softmax_2d_x
    expected = log_softmax_2d_expected
    assert_allclose(sc.log_softmax(x, axis=1), expected, rtol=1e-13)