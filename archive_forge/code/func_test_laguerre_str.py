from math import nan, inf
import pytest
from numpy.core import array, arange, printoptions
import numpy.polynomial as poly
from numpy.testing import assert_equal, assert_
from fractions import Fraction
from decimal import Decimal
@pytest.mark.parametrize(('inp', 'tgt'), (([1, 2, 3], '1.0 + 2.0 L_1(x) + 3.0 L_2(x)'), ([-1, 0, 3, -1], '-1.0 + 0.0 L_1(x) + 3.0 L_2(x) - 1.0 L_3(x)'), (arange(12), '0.0 + 1.0 L_1(x) + 2.0 L_2(x) + 3.0 L_3(x) + 4.0 L_4(x) + 5.0 L_5(x) +\n6.0 L_6(x) + 7.0 L_7(x) + 8.0 L_8(x) + 9.0 L_9(x) + 10.0 L_10(x) +\n11.0 L_11(x)')))
def test_laguerre_str(self, inp, tgt):
    res = str(poly.Laguerre(inp))
    assert_equal(res, tgt)