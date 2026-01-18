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
def test_FourierSeries__operations():
    fo, fe, fp = _get_examples()
    fes = fe.scale(-1).shift(pi ** 2)
    assert fes.truncate() == 4 * cos(x) - cos(2 * x) + 2 * pi ** 2 / 3
    assert fp.shift(-pi / 2).truncate() == 2 * sin(x) + 2 * sin(3 * x) / 3 + 2 * sin(5 * x) / 5
    fos = fo.scale(3)
    assert fos.truncate() == 6 * sin(x) - 3 * sin(2 * x) + 2 * sin(3 * x)
    fx = fe.scalex(2).shiftx(1)
    assert fx.truncate() == -4 * cos(2 * x + 2) + cos(4 * x + 4) + pi ** 2 / 3
    fl = fe.scalex(3).shift(-pi).scalex(2).shiftx(1).scale(4)
    assert fl.truncate() == -16 * cos(6 * x + 6) + 4 * cos(12 * x + 12) - 4 * pi + 4 * pi ** 2 / 3
    raises(ValueError, lambda: fo.shift(x))
    raises(ValueError, lambda: fo.shiftx(sin(x)))
    raises(ValueError, lambda: fo.scale(x * y))
    raises(ValueError, lambda: fo.scalex(x ** 2))