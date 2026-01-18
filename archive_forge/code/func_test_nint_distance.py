import mpmath
from mpmath import *
from mpmath.libmp import *
import random
import sys
def test_nint_distance():
    assert nint_distance(mpf(-3)) == (-3, -inf)
    assert nint_distance(mpc(-3)) == (-3, -inf)
    assert nint_distance(mpf(-3.1)) == (-3, -3)
    assert nint_distance(mpf(-3.01)) == (-3, -6)
    assert nint_distance(mpf(-3.001)) == (-3, -9)
    assert nint_distance(mpf(-3.0001)) == (-3, -13)
    assert nint_distance(mpf(-2.9)) == (-3, -3)
    assert nint_distance(mpf(-2.99)) == (-3, -6)
    assert nint_distance(mpf(-2.999)) == (-3, -9)
    assert nint_distance(mpf(-2.9999)) == (-3, -13)
    assert nint_distance(mpc(-3 + 0.1j)) == (-3, -3)
    assert nint_distance(mpc(-3 + 0.01j)) == (-3, -6)
    assert nint_distance(mpc(-3.1 + 0.1j)) == (-3, -3)
    assert nint_distance(mpc(-3.01 + 0.01j)) == (-3, -6)
    assert nint_distance(mpc(-3.001 + 0.001j)) == (-3, -9)
    assert nint_distance(mpf(0)) == (0, -inf)
    assert nint_distance(mpf(0.01)) == (0, -6)
    assert nint_distance(mpf('1e-100')) == (0, -332)