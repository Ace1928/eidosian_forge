import pytest
from mpmath import *
def test_double_3():
    assert ae(quadts(lambda x, y: 1 / sqrt(1 + x * x + y * y), [-1, 1], [-1, 1]), 4 * log(2 + sqrt(3)) - 2 * pi / 3)