from sympy.core.add import Add
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin, sinc, tan)
from sympy.series.fourier import fourier_series
from sympy.series.fourier import FourierSeries
from sympy.testing.pytest import raises
from functools import lru_cache
def test_FourierSeries():
    fo, fe, fp = _get_examples()
    assert fourier_series(1, (-pi, pi)) == 1
    assert Piecewise((0, x < 0), (pi, True)).fourier_series((x, -pi, pi)).truncate() == fp.truncate()
    assert isinstance(fo, FourierSeries)
    assert fo.function == x
    assert fo.x == x
    assert fo.period == (-pi, pi)
    assert fo.term(3) == 2 * sin(3 * x) / 3
    assert fe.term(3) == -4 * cos(3 * x) / 9
    assert fp.term(3) == 2 * sin(3 * x) / 3
    assert fo.as_leading_term(x) == 2 * sin(x)
    assert fe.as_leading_term(x) == pi ** 2 / 3
    assert fp.as_leading_term(x) == pi / 2
    assert fo.truncate() == 2 * sin(x) - sin(2 * x) + 2 * sin(3 * x) / 3
    assert fe.truncate() == -4 * cos(x) + cos(2 * x) + pi ** 2 / 3
    assert fp.truncate() == 2 * sin(x) + 2 * sin(3 * x) / 3 + pi / 2
    fot = fo.truncate(n=None)
    s = [0, 2 * sin(x), -sin(2 * x)]
    for i, t in enumerate(fot):
        if i == 3:
            break
        assert s[i] == t

    def _check_iter(f, i):
        for ind, t in enumerate(f):
            assert t == f[ind]
            if ind == i:
                break
    _check_iter(fo, 3)
    _check_iter(fe, 3)
    _check_iter(fp, 3)
    assert fo.subs(x, x ** 2) == fo
    raises(ValueError, lambda: fourier_series(x, (0, 1, 2)))
    raises(ValueError, lambda: fourier_series(x, (x, 0, oo)))
    raises(ValueError, lambda: fourier_series(x * y, (0, oo)))