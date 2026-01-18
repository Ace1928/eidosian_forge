import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_almost_equals_2(self):
    EPSILON = 1e-10
    E = EPSILON * 0.5
    t = Affine(1.0, E, 0, -E, 1.0 + E, E)
    assert t.almost_equals(Affine.identity(), precision=EPSILON)
    assert Affine.identity().almost_equals(t, precision=EPSILON)
    assert t.almost_equals(t, precision=EPSILON)
    t = Affine(1.0, 0, 0, -EPSILON, 1.0, 0)
    assert not t.almost_equals(Affine.identity(), precision=EPSILON)
    assert not Affine.identity().almost_equals(t, precision=EPSILON)
    assert t.almost_equals(t, precision=EPSILON)