import pytest
import numpy.polynomial as poly
from numpy.core import array
from numpy.testing import assert_equal, assert_raises, assert_
def test_froomroots():
    roots = [-2, 2]
    p = poly.Polynomial.fromroots(roots, symbol='z')
    assert_equal(p.symbol, 'z')