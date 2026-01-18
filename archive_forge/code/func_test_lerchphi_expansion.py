from sympy.concrete.summations import Sum
from sympy.core.function import expand_func
from sympy.core.numbers import (Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import (Abs, polar_lift)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.zeta_functions import (dirichlet_eta, lerchphi, polylog, riemann_xi, stieltjes, zeta)
from sympy.series.order import O
from sympy.core.function import ArgumentIndexError
from sympy.functions.combinatorial.numbers import bernoulli, factorial, genocchi, harmonic
from sympy.testing.pytest import raises
from sympy.core.random import (test_derivative_numerically as td,
def test_lerchphi_expansion():
    assert myexpand(lerchphi(1, s, a), zeta(s, a))
    assert myexpand(lerchphi(z, s, 1), polylog(s, z) / z)
    assert myexpand(lerchphi(z, -1, a), a / (1 - z) + z / (1 - z) ** 2)
    assert myexpand(lerchphi(z, -3, a), None)
    assert myexpand(lerchphi(z, s, S.Half), 2 ** (s - 1) * (polylog(s, sqrt(z)) / sqrt(z) - polylog(s, polar_lift(-1) * sqrt(z)) / sqrt(z)))
    assert myexpand(lerchphi(z, s, 2), -1 / z + polylog(s, z) / z ** 2)
    assert myexpand(lerchphi(z, s, Rational(3, 2)), None)
    assert myexpand(lerchphi(z, s, Rational(7, 3)), None)
    assert myexpand(lerchphi(z, s, Rational(-1, 3)), None)
    assert myexpand(lerchphi(z, s, Rational(-5, 2)), None)
    assert myexpand(lerchphi(-1, s, a), 2 ** (-s) * zeta(s, a / 2) - 2 ** (-s) * zeta(s, (a + 1) / 2))
    assert myexpand(lerchphi(I, s, a), None)
    assert myexpand(lerchphi(-I, s, a), None)
    assert myexpand(lerchphi(exp(I * pi * Rational(2, 5)), s, a), None)