import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_mul_fallback_unpack():
    """Support fallback in case that other is a single object."""

    class TextPoint:
        """Not iterable, will trigger ValueError in Affine.__mul__."""

        def __rmul__(self, other):
            return other * (1, 2)
    assert Affine.identity() * TextPoint() == (1, 2)