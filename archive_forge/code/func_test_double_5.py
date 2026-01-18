import pytest
from mpmath import *
def test_double_5():
    assert ae(quadts(lambda x, y: 1 / (1 - x * y), [0, 1], [0, 1]), pi ** 2 / 6)