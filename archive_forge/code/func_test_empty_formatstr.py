from math import nan, inf
import pytest
from numpy.core import array, arange, printoptions
import numpy.polynomial as poly
from numpy.testing import assert_equal, assert_
from fractions import Fraction
from decimal import Decimal
def test_empty_formatstr(self):
    poly.set_default_printstyle('ascii')
    p = poly.Polynomial([1, 2, 3])
    assert_equal(format(p), '1.0 + 2.0 x + 3.0 x**2')
    assert_equal(f'{p}', '1.0 + 2.0 x + 3.0 x**2')