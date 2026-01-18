import numpy as np
from numpy.testing import assert_allclose
import pytest
import scipy.special as sc
def test_log_softmax_noneaxis(log_softmax_x, log_softmax_expected):
    x = log_softmax_x.reshape(2, 2)
    expected = log_softmax_expected.reshape(2, 2)
    assert_allclose(sc.log_softmax(x), expected, rtol=1e-13)