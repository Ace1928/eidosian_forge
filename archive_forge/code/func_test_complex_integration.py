import pytest
from mpmath import *
def test_complex_integration():
    assert quadts(lambda x: x, [0, 1 + j]).ae(j)