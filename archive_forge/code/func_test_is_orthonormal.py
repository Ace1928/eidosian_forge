import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_is_orthonormal(self):
    assert Affine.identity().is_orthonormal
    assert Affine.translation(4, -1).is_orthonormal
    assert Affine.rotation(90).is_orthonormal
    assert Affine.rotation(-26).is_orthonormal
    assert not Affine.scale(2.5, 6.1).is_orthonormal
    assert not Affine.scale(0.5, 2).is_orthonormal
    assert not Affine.shear(4, -1).is_orthonormal