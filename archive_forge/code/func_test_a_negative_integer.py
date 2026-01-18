import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import scipy.special as sc
@pytest.mark.parametrize('a, b, x, result', [(-1, 1, 1.5, -0.5), (-10, 1, 1.5, 0.4180177743094308), (-25, 1, 1.5, 0.2511449164603784), (-50, 1, 1.5, -0.2568364397519476), (-80, 1, 1.5, -0.24554329325751503), (-150, 1, 1.5, -0.17336479551542044)])
def test_a_negative_integer(self, a, b, x, result):
    assert_allclose(sc.hyp1f1(a, b, x), result, atol=0, rtol=2e-14)