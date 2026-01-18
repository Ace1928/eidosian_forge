import pytest
import numpy.polynomial as poly
from numpy.core import array
from numpy.testing import assert_equal, assert_raises, assert_
def test_change_symbol(self):
    p = poly.Polynomial(self.c, symbol='y')
    pt = poly.Polynomial(p.coef, symbol='t')
    assert_equal(pt.symbol, 't')