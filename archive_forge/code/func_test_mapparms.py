import operator as op
from numbers import Number
import pytest
import numpy as np
from numpy.polynomial import (
from numpy.testing import (
from numpy.polynomial.polyutils import RankWarning
def test_mapparms(Poly):
    d = Poly.domain
    w = Poly.window
    p = Poly([1], domain=d, window=w)
    assert_almost_equal([0, 1], p.mapparms())
    w = 2 * d + 1
    p = Poly([1], domain=d, window=w)
    assert_almost_equal([1, 2], p.mapparms())