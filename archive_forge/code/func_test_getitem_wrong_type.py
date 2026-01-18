import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_getitem_wrong_type(self):
    t = Affine(1, 2, 3, 4, 5, 6)
    with pytest.raises(TypeError):
        t['foobar']