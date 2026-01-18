import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_imul_transform(self):
    t = Affine.translation(3, 5)
    t *= Affine.translation(-2, 3.5)
    assert isinstance(t, Affine)
    seq_almost_equal(t, Affine.translation(1, 8.5))