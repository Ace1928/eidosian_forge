from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, oo)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.series.order import O
from sympy.sets.sets import Interval
from sympy.simplify.radsimp import collect
from sympy.simplify.simplify import simplify
from sympy.core.exprtools import (decompose_power, Factors, Term, _gcd_terms,
from sympy.core.mul import _keep_coeff as _keep_coeff
from sympy.simplify.cse_opts import sub_pre
from sympy.testing.pytest import raises
from sympy.abc import a, b, t, x, y, z
def test_factor_nc():
    x, y = symbols('x,y')
    k = symbols('k', integer=True)
    n, m, o = symbols('n,m,o', commutative=False)
    from sympy.core.function import _mexpand
    e = x * (1 + y) ** 2
    assert _mexpand(e) == x + x * 2 * y + x * y ** 2

    def factor_nc_test(e):
        ex = _mexpand(e)
        assert ex.is_Add
        f = factor_nc(ex)
        assert not f.is_Add and _mexpand(f) == ex
    factor_nc_test(x * (1 + y))
    factor_nc_test(n * (x + 1))
    factor_nc_test(n * (x + m))
    factor_nc_test((x + m) * n)
    factor_nc_test(n * m * (x * o + n * o * m) * n)
    s = Sum(x, (x, 1, 2))
    factor_nc_test(x * (1 + s))
    factor_nc_test(x * (1 + s) * s)
    factor_nc_test(x * (1 + sin(s)))
    factor_nc_test((1 + n) ** 2)
    factor_nc_test((x + n) * (x + m) * (x + y))
    factor_nc_test(x * (n * m + 1))
    factor_nc_test(x * (n * m + x))
    factor_nc_test(x * (x * n * m + 1))
    factor_nc_test(n * (m / x + o))
    factor_nc_test(m * (n + o / 2))
    factor_nc_test(x * n * (x * m + 1))
    factor_nc_test(x * (m * n + x * n * m))
    factor_nc_test(n * (1 - m) * n ** 2)
    factor_nc_test((n + m) ** 2)
    factor_nc_test((n - m) * (n + m) ** 2)
    factor_nc_test((n + m) ** 2 * (n - m))
    factor_nc_test((m - n) * (n + m) ** 2 * (n - m))
    assert factor_nc(n * (n + n * m)) == n ** 2 * (1 + m)
    assert factor_nc(m * (m * n + n * m * n ** 2)) == m * (m + n * m * n) * n
    eq = m * sin(n) - sin(n) * m
    assert factor_nc(eq) == eq
    from sympy.physics.secondquant import Commutator
    from sympy.polys.polytools import factor
    eq = 1 + x * Commutator(m, n)
    assert factor_nc(eq) == eq
    eq = x * Commutator(m, n) + x * Commutator(m, o) * Commutator(m, n)
    assert factor(eq) == x * (1 + Commutator(m, o)) * Commutator(m, n)
    assert (2 * n + 2 * m).factor() == 2 * (n + m)
    _n = symbols('nz', zero=False, commutative=False)
    assert factor_nc(_n ** k + _n ** (k + 1)) == _n ** k * (1 + _n)
    assert factor_nc((m * n) ** k + (m * n) ** (k + 1)) == (1 + m * n) * (m * n) ** k
    assert factor_nc(-n * (2 * x ** 2 + 2 * x)) == -2 * n * x * (x + 1)