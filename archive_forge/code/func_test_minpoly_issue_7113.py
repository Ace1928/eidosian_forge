from sympy.core.function import expand
from sympy.core import (GoldenRatio, TribonacciConstant)
from sympy.core.numbers import (AlgebraicNumber, I, Rational, oo, pi)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import (cbrt, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import CRootOf
from sympy.solvers.solveset import nonlinsolve
from sympy.geometry import Circle, intersection
from sympy.testing.pytest import raises, slow
from sympy.sets.sets import FiniteSet
from sympy.geometry.point import Point2D
from sympy.polys.numberfields.minpoly import (
from sympy.polys.partfrac import apart
from sympy.polys.polyerrors import (
from sympy.polys.domains import QQ
from sympy.polys.rootoftools import rootof
from sympy.polys.polytools import degree
from sympy.abc import x, y, z
def test_minpoly_issue_7113():
    from sympy.simplify.simplify import nsimplify
    r = nsimplify(pi, tolerance=1e-09)
    mp = minimal_polynomial(r, x)
    assert mp == 1768292677839237920489538677417507171630859375 * x ** 109 - 2734577732179183863586489182929671773182898498218854181690460140337930774573792597743853652058046464