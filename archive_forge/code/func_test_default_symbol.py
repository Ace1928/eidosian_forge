import pytest
import numpy.polynomial as poly
from numpy.core import array
from numpy.testing import assert_equal, assert_raises, assert_
def test_default_symbol(self):
    p = poly.Polynomial(self.c)
    assert_equal(p.symbol, 'x')