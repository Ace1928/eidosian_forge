import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_bad_value_world(self):
    """Wrong number of parameters."""
    with pytest.raises(ValueError):
        affine.loadsw('1.0\n0.0\n0.0\n1.0\n0.0\n0.0\n0.0')