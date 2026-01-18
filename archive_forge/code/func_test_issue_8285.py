from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.functions.elementary.complexes import (conjugate, im, re)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, cos, sin)
from sympy.polys.domains.integerring import ZZ
from sympy.sets.sets import Interval
from sympy.simplify.powsimp import powsimp
from sympy.polys import Poly, cyclotomic_poly, intervals, nroots, rootof
from sympy.polys.polyroots import (root_factors, roots_linear,
from sympy.polys.orthopolys import legendre_poly
from sympy.polys.polyerrors import PolynomialError, \
from sympy.polys.polyutils import _nsort
from sympy.testing.pytest import raises, slow
from sympy.core.random import verify_numerically
import mpmath
from itertools import product
def test_issue_8285():
    roots = (Poly(4 * x ** 8 - 1, x) * Poly(x ** 2 + 1)).all_roots()
    assert _check(roots)
    f = Poly(x ** 4 + 5 * x ** 2 + 6, x)
    ro = [rootof(f, i) for i in range(4)]
    roots = Poly(x ** 4 + 5 * x ** 2 + 6, x).all_roots()
    assert roots == ro
    assert _check(roots)
    roots = Poly(2 * x ** 8 - 1).all_roots()
    assert _check(roots)
    assert len(Poly(2 * x ** 10 - 1).all_roots()) == 10