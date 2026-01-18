import pytest
from mpmath import *
def test_double_trivial():
    assert ae(quadts(lambda x, y: x, [0, 1], [0, 1]), 0.5)
    assert ae(quadts(lambda x, y: x, [-1, 1], [-1, 1]), 0.0)