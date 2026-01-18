from sympy.core.numbers import (I, Rational, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.elementary.hyperbolic import atanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (sin, tan)
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.integrals.integrals import Integral
from sympy.series.order import O
from sympy.functions.special.elliptic_integrals import (elliptic_k as K,
from sympy.core.random import (test_derivative_numerically as td,
from sympy.abc import z, m, n
def test_P():
    assert P(0, z, m) == F(z, m)
    assert P(1, z, m) == F(z, m) + (sqrt(1 - m * sin(z) ** 2) * tan(z) - E(z, m)) / (1 - m)
    assert P(n, i * pi / 2, m) == i * P(n, m)
    assert P(n, z, 0) == atanh(sqrt(n - 1) * tan(z)) / sqrt(n - 1)
    assert P(n, z, n) == F(z, n) - P(1, z, n) + tan(z) / sqrt(1 - n * sin(z) ** 2)
    assert P(oo, z, m) == 0
    assert P(-oo, z, m) == 0
    assert P(n, z, oo) == 0
    assert P(n, z, -oo) == 0
    assert P(0, m) == K(m)
    assert P(1, m) is zoo
    assert P(n, 0) == pi / (2 * sqrt(1 - n))
    assert P(2, 1) is -oo
    assert P(-1, 1) is oo
    assert P(n, n) == E(n) / (1 - n)
    assert P(n, -z, m) == -P(n, z, m)
    ni, mi = (Symbol('n', real=False), Symbol('m', real=False))
    assert P(ni, z, mi).conjugate() == P(ni.conjugate(), z.conjugate(), mi.conjugate())
    nr, mr = (Symbol('n', negative=True), Symbol('m', negative=True))
    assert P(nr, z, mr).conjugate() == P(nr, z.conjugate(), mr)
    assert P(n, m).conjugate() == P(n.conjugate(), m.conjugate())
    assert P(n, z, m).diff(n) == (E(z, m) + (m - n) * F(z, m) / n + (n ** 2 - m) * P(n, z, m) / n - n * sqrt(1 - m * sin(z) ** 2) * sin(2 * z) / (2 * (1 - n * sin(z) ** 2))) / (2 * (m - n) * (n - 1))
    assert P(n, z, m).diff(z) == 1 / (sqrt(1 - m * sin(z) ** 2) * (1 - n * sin(z) ** 2))
    assert P(n, z, m).diff(m) == (E(z, m) / (m - 1) + P(n, z, m) - m * sin(2 * z) / (2 * (m - 1) * sqrt(1 - m * sin(z) ** 2))) / (2 * (n - m))
    assert P(n, m).diff(n) == (E(m) + (m - n) * K(m) / n + (n ** 2 - m) * P(n, m) / n) / (2 * (m - n) * (n - 1))
    assert P(n, m).diff(m) == (E(m) / (m - 1) + P(n, m)) / (2 * (n - m))
    assert P(n, z, m).series(z) == z + z ** 3 * (m / 6 + n / 3) + z ** 5 * (3 * m ** 2 / 40 + m * n / 10 - m / 30 + n ** 2 / 5 - n / 15) + O(z ** 6)
    assert P(n, z, m).rewrite(Integral).dummy_eq(Integral(1 / ((1 - n * sin(t) ** 2) * sqrt(1 - m * sin(t) ** 2)), (t, 0, z)))
    assert P(n, m).rewrite(Integral).dummy_eq(Integral(1 / ((1 - n * sin(t) ** 2) * sqrt(1 - m * sin(t) ** 2)), (t, 0, pi / 2)))