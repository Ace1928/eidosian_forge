from math import nan, inf
import pytest
from numpy.core import array, arange, printoptions
import numpy.polynomial as poly
from numpy.testing import assert_equal, assert_
from fractions import Fraction
from decimal import Decimal
@pytest.mark.parametrize(('inp', 'tgt'), (([1, 2, 3], '1.0 + 2.0 He_1(x) + 3.0 He_2(x)'), ([-1, 0, 3, -1], '-1.0 + 0.0 He_1(x) + 3.0 He_2(x) - 1.0 He_3(x)'), (arange(12), '0.0 + 1.0 He_1(x) + 2.0 He_2(x) + 3.0 He_3(x) + 4.0 He_4(x) +\n5.0 He_5(x) + 6.0 He_6(x) + 7.0 He_7(x) + 8.0 He_8(x) + 9.0 He_9(x) +\n10.0 He_10(x) + 11.0 He_11(x)')))
def test_hermiteE_str(self, inp, tgt):
    res = str(poly.HermiteE(inp))
    assert_equal(res, tgt)