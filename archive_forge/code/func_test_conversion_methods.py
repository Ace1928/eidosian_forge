import random
from mpmath import *
from mpmath.libmp import *
def test_conversion_methods():

    class SomethingRandom:
        pass

    class SomethingReal:

        def _mpmath_(self, prec, rounding):
            return mp.make_mpf(from_str('1.3', prec, rounding))

    class SomethingComplex:

        def _mpmath_(self, prec, rounding):
            return mp.make_mpc((from_str('1.3', prec, rounding), from_str('1.7', prec, rounding)))
    x = mpf(3)
    z = mpc(3)
    a = SomethingRandom()
    y = SomethingReal()
    w = SomethingComplex()
    for d in [15, 45]:
        mp.dps = d
        assert (x + y).ae(mpf('4.3'))
        assert (y + x).ae(mpf('4.3'))
        assert (x + w).ae(mpc('4.3', '1.7'))
        assert (w + x).ae(mpc('4.3', '1.7'))
        assert (z + y).ae(mpc('4.3'))
        assert (y + z).ae(mpc('4.3'))
        assert (z + w).ae(mpc('4.3', '1.7'))
        assert (w + z).ae(mpc('4.3', '1.7'))
        x - y
        y - x
        x - w
        w - x
        z - y
        y - z
        z - w
        w - z
        x * y
        y * x
        x * w
        w * x
        z * y
        y * z
        z * w
        w * z
        x / y
        y / x
        x / w
        w / x
        z / y
        y / z
        z / w
        w / z
        x ** y
        y ** x
        x ** w
        w ** x
        z ** y
        y ** z
        z ** w
        w ** z
        x == y
        y == x
        x == w
        w == x
        z == y
        y == z
        z == w
        w == z
    mp.dps = 15
    assert x.__add__(a) is NotImplemented
    assert x.__radd__(a) is NotImplemented
    assert x.__lt__(a) is NotImplemented
    assert x.__gt__(a) is NotImplemented
    assert x.__le__(a) is NotImplemented
    assert x.__ge__(a) is NotImplemented
    assert x.__eq__(a) is NotImplemented
    assert x.__ne__(a) is NotImplemented
    if hasattr(x, '__cmp__'):
        assert x.__cmp__(a) is NotImplemented
    assert x.__sub__(a) is NotImplemented
    assert x.__rsub__(a) is NotImplemented
    assert x.__mul__(a) is NotImplemented
    assert x.__rmul__(a) is NotImplemented
    assert x.__div__(a) is NotImplemented
    assert x.__rdiv__(a) is NotImplemented
    assert x.__mod__(a) is NotImplemented
    assert x.__rmod__(a) is NotImplemented
    assert x.__pow__(a) is NotImplemented
    assert x.__rpow__(a) is NotImplemented
    assert z.__add__(a) is NotImplemented
    assert z.__radd__(a) is NotImplemented
    assert z.__eq__(a) is NotImplemented
    assert z.__ne__(a) is NotImplemented
    assert z.__sub__(a) is NotImplemented
    assert z.__rsub__(a) is NotImplemented
    assert z.__mul__(a) is NotImplemented
    assert z.__rmul__(a) is NotImplemented
    assert z.__div__(a) is NotImplemented
    assert z.__rdiv__(a) is NotImplemented
    assert z.__pow__(a) is NotImplemented
    assert z.__rpow__(a) is NotImplemented