from sympy.integrals.risch import DifferentialExtension, derivation
from sympy.integrals.prde import (prde_normal_denom, prde_special_denom,
from sympy.polys.polymatrix import PolyMatrix as Matrix
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.polytools import Poly
from sympy.abc import x, t, n
def test_is_deriv_k():
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 / x, t1), Poly(1 / (x + 1), t2)], 'exts': [None, 'log', 'log'], 'extargs': [None, x, x + 1]})
    assert is_deriv_k(Poly(2 * x ** 2 + 2 * x, t2), Poly(1, t2), DE) == ([(t1, 1), (t2, 1)], t1 + t2, 2)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 / x, t1), Poly(t2, t2)], 'exts': [None, 'log', 'exp'], 'extargs': [None, x, x]})
    assert is_deriv_k(Poly(x ** 2 * t2 ** 3, t2), Poly(1, t2), DE) == ([(x, 3), (t1, 2)], 2 * t1 + 3 * x, 1)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(2 / x, t1)], 'exts': [None, 'log'], 'extargs': [None, x ** 2]})
    assert is_deriv_k(Poly(x, t1), Poly(1, t1), DE) == ([(t1, S.Half)], t1 / 2, 1)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(2 / (1 + x), t0)], 'exts': [None, 'log'], 'extargs': [None, x ** 2 + 2 * x + 1]})
    assert is_deriv_k(Poly(1 + x, t0), Poly(1, t0), DE) == ([(t0, S.Half)], t0 / 2, 1)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(-1 / x, t)], 'exts': [None, 'log'], 'extargs': [None, 1 / x]})
    assert is_deriv_k(Poly(1, t), Poly(x, t), DE) == ([(t, 1)], t, 1)