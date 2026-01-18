import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_column_vectors(self):
    a, b, c = Affine(2, 3, 4, 5, 6, 7).column_vectors
    assert isinstance(a, tuple)
    assert isinstance(b, tuple)
    assert isinstance(c, tuple)
    assert a == (2, 5)
    assert b == (3, 6)
    assert c == (4, 7)