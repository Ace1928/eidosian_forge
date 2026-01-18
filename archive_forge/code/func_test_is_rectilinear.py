import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_is_rectilinear(self):
    assert Affine.identity().is_rectilinear
    assert Affine.scale(2.5, 6.1).is_rectilinear
    assert Affine.translation(4, -1).is_rectilinear
    assert Affine.rotation(90).is_rectilinear
    assert not Affine.shear(4, -1).is_rectilinear
    assert not Affine.rotation(-26).is_rectilinear