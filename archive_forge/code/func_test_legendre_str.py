from math import nan, inf
import pytest
from numpy.core import array, arange, printoptions
import numpy.polynomial as poly
from numpy.testing import assert_equal, assert_
from fractions import Fraction
from decimal import Decimal
@pytest.mark.parametrize(('inp', 'tgt'), (([1, 2, 3], '1.0 + 2.0 P_1(x) + 3.0 P_2(x)'), ([-1, 0, 3, -1], '-1.0 + 0.0 P_1(x) + 3.0 P_2(x) - 1.0 P_3(x)'), (arange(12), '0.0 + 1.0 P_1(x) + 2.0 P_2(x) + 3.0 P_3(x) + 4.0 P_4(x) + 5.0 P_5(x) +\n6.0 P_6(x) + 7.0 P_7(x) + 8.0 P_8(x) + 9.0 P_9(x) + 10.0 P_10(x) +\n11.0 P_11(x)')))
def test_legendre_str(self, inp, tgt):
    res = str(poly.Legendre(inp))
    assert_equal(res, tgt)