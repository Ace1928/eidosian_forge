import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_determinant(self):
    assert Affine.identity().determinant == 1
    assert Affine.scale(2).determinant == 4
    assert Affine.scale(0).determinant == 0
    assert Affine.scale(5, 1).determinant == 5
    assert Affine.scale(-1, 1).determinant == -1
    assert Affine.scale(-1, 0).determinant == 0
    assert Affine.rotation(77).determinant == pytest.approx(1)
    assert Affine.translation(32, -47).determinant == pytest.approx(1)