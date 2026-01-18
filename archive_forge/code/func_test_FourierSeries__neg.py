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
def test_FourierSeries__neg():
    fo, fe, fp = _get_examples()
    assert (-fo).truncate() == -2 * sin(x) + sin(2 * x) - 2 * sin(3 * x) / 3
    assert (-fe).truncate() == +4 * cos(x) - cos(2 * x) - pi ** 2 / 3