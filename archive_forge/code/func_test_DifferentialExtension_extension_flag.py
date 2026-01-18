from sympy.core.function import (Function, Lambda, diff, expand_log)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan, sin, tan)
from sympy.polys.polytools import (Poly, cancel, factor)
from sympy.integrals.risch import (gcdex_diophantine, frac_in, as_poly_1t,
from sympy.testing.pytest import raises
from sympy.abc import x, t, nu, z, a, y
def test_DifferentialExtension_extension_flag():
    raises(ValueError, lambda: DifferentialExtension(extension={'T': [x, t]}))
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    assert DE._important_attrs == (None, None, [Poly(1, x), Poly(t, t)], [x, t], None, None, None, None)
    assert DE.d == Poly(t, t)
    assert DE.t == t
    assert DE.level == -1
    assert DE.cases == ['base', 'exp']
    assert DE.x == x
    assert DE.case == 'exp'
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)], 'exts': [None, 'exp'], 'extargs': [None, x]})
    assert DE._important_attrs == (None, None, [Poly(1, x), Poly(t, t)], [x, t], None, None, [None, 'exp'], [None, x])
    raises(ValueError, lambda: DifferentialExtension())