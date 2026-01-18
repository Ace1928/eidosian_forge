import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_mul_wrong_type(self):
    with pytest.raises(TypeError):
        Affine(1, 2, 3, 4, 5, 6) * None