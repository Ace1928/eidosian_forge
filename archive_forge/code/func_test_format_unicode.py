from math import nan, inf
import pytest
from numpy.core import array, arange, printoptions
import numpy.polynomial as poly
from numpy.testing import assert_equal, assert_
from fractions import Fraction
from decimal import Decimal
def test_format_unicode(self):
    poly.set_default_printstyle('ascii')
    p = poly.Polynomial([1, 2, 0, -1])
    assert_equal(format(p, 'unicode'), '1.0 + 2.0·x + 0.0·x² - 1.0·x³')