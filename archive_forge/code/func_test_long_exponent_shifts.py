from mpmath import *
from mpmath.libmp import *
def test_long_exponent_shifts():
    mp.dps = 15
    x = mpf(1)
    for p in [32, 64]:
        a = ldexp(1, 2 ** (p - 1))
        b = ldexp(1, 2 ** p)
        c = ldexp(1, 2 ** (p + 1))
        d = ldexp(1, -2 ** (p - 1))
        e = ldexp(1, -2 ** p)
        f = ldexp(1, -2 ** (p + 1))
        assert x + a == a
        assert x + b == b
        assert x + c == c
        assert x + d == x
        assert x + e == x
        assert x + f == x
        assert a + x == a
        assert b + x == b
        assert c + x == c
        assert d + x == x
        assert e + x == x
        assert f + x == x
        assert x - a == -a
        assert x - b == -b
        assert x - c == -c
        assert x - d == x
        assert x - e == x
        assert x - f == x
        assert a - x == a
        assert b - x == b
        assert c - x == c
        assert d - x == -x
        assert e - x == -x
        assert f - x == -x