import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_mul_fallback_type_error():
    """Support fallback in case that other is an unexpected type."""

    class TextPoint:
        """Iterable, but values trigger TypeError in Affine.__mul__."""

        def __iter__(self):
            return ('1', '2')

        def __rmul__(self, other):
            return other * (1, 2)
    assert Affine.identity() * TextPoint() == (1, 2)