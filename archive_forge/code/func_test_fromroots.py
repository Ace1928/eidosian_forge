import operator as op
from numbers import Number
import pytest
import numpy as np
from numpy.polynomial import (
from numpy.testing import (
from numpy.polynomial.polyutils import RankWarning
def test_fromroots(Poly):
    d = Poly.domain + random((2,)) * 0.25
    w = Poly.window + random((2,)) * 0.25
    r = random((5,))
    p1 = Poly.fromroots(r, domain=d, window=w)
    assert_equal(p1.degree(), len(r))
    assert_equal(p1.domain, d)
    assert_equal(p1.window, w)
    assert_almost_equal(p1(r), 0)
    pdom = Polynomial.domain
    pwin = Polynomial.window
    p2 = Polynomial.cast(p1, domain=pdom, window=pwin)
    assert_almost_equal(p2.coef[-1], 1)