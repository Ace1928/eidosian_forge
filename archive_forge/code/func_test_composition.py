import pytest
import numpy.polynomial as poly
from numpy.core import array
from numpy.testing import assert_equal, assert_raises, assert_
def test_composition():
    p = poly.Polynomial([3, 2, 1], symbol='t')
    q = poly.Polynomial([5, 1, 0, -1], symbol='λ_1')
    r = p(q)
    assert r.symbol == 'λ_1'