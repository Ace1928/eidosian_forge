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
def test_FourierSeries_finite():
    assert fourier_series(sin(x)).truncate(1) == sin(x)
    assert fourier_series(sin(x) * log(y) * exp(z), (x, pi, -pi)).truncate() == sin(x) * log(y) * exp(z)
    assert fourier_series(sin(x) ** 6).truncate(oo) == -15 * cos(2 * x) / 32 + 3 * cos(4 * x) / 16 - cos(6 * x) / 32 + Rational(5, 16)
    assert fourier_series(sin(x) ** 6).truncate() == -15 * cos(2 * x) / 32 + 3 * cos(4 * x) / 16 + Rational(5, 16)
    assert fourier_series(sin(4 * x + 3) + cos(3 * x + 4)).truncate(oo) == -sin(4) * sin(3 * x) + sin(4 * x) * cos(3) + cos(4) * cos(3 * x) + sin(3) * cos(4 * x)
    assert fourier_series(sin(x) + cos(x) * tan(x)).truncate(oo) == 2 * sin(x)
    assert fourier_series(cos(pi * x), (x, -1, 1)).truncate(oo) == cos(pi * x)
    assert fourier_series(cos(3 * pi * x + 4) - sin(4 * pi * x) * log(pi * y), (x, -1, 1)).truncate(oo) == -log(pi * y) * sin(4 * pi * x) - sin(4) * sin(3 * pi * x) + cos(4) * cos(3 * pi * x)