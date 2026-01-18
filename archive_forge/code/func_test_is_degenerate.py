import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_is_degenerate(self):
    assert not Affine.identity().is_degenerate
    assert not Affine.translation(2, -1).is_degenerate
    assert not Affine.shear(0, -22.5).is_degenerate
    assert not Affine.rotation(88.7).is_degenerate
    assert not Affine.scale(0.5).is_degenerate
    assert Affine.scale(0).is_degenerate
    assert Affine.scale(-10, 0).is_degenerate
    assert Affine.scale(0, 300).is_degenerate
    assert Affine.scale(0).is_degenerate
    assert Affine.scale(0).is_degenerate