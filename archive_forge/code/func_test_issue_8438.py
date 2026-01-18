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
def test_issue_8438():
    p = Poly([1, y, -2, -3], x).as_expr()
    roots = roots_cubic(Poly(p, x), x)
    z = Rational(-3, 2) - I * 7 / 2
    post = [r.subs(y, z) for r in roots]
    assert set(post) == set(roots_cubic(Poly(p.subs(y, z), x)))
    assert all((p.subs({y: z, x: i}).n(2, chop=True) == 0 for i in post))