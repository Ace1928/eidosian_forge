import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import scipy.special as sc
def test_a_neg_int_and_b_equal_x(self):
    a = -10.0
    b = 2.5
    x = 2.5
    expected = 0.03653236643641043
    computed = sc.hyp1f1(a, b, x)
    assert_allclose(computed, expected, atol=0, rtol=1e-13)