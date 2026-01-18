from mpmath import *
from mpmath.libmp import *
def test_bitcount():
    assert bitcount(0) == 0
    assert bitcount(1) == 1
    assert bitcount(7) == 3
    assert bitcount(8) == 4
    assert bitcount(2 ** 100) == 101
    assert bitcount(2 ** 100 - 1) == 100