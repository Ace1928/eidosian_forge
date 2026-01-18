import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_args_too_many(self):
    with pytest.raises(TypeError):
        Affine(*range(10))