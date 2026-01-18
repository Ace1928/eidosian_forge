import numpy as np
from numpy.testing import assert_allclose
import pytest
import scipy.special as sc
def test_log_softmax_translation(log_softmax_x, log_softmax_expected):
    x = log_softmax_x + 100
    expected = log_softmax_expected
    assert_allclose(sc.log_softmax(x), expected, rtol=1e-13)