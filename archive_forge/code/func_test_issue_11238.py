import random
import concurrent.futures
from collections.abc import Hashable
from sympy.core.add import Add
from sympy.core.function import (Function, diff, expand)
from sympy.core.numbers import (E, Float, I, Integer, Rational, nan, oo, pi)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.integrals.integrals import integrate
from sympy.polys.polytools import (Poly, PurePoly)
from sympy.printing.str import sstr
from sympy.sets.sets import FiniteSet
from sympy.simplify.simplify import (signsimp, simplify)
from sympy.simplify.trigsimp import trigsimp
from sympy.matrices.matrices import (ShapeError, MatrixError,
from sympy.matrices import (
from sympy.matrices.utilities import _dotprodsimp_state
from sympy.core import Tuple, Wild
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.utilities.iterables import flatten, capture, iterable
from sympy.utilities.exceptions import ignore_warnings, SymPyDeprecationWarning
from sympy.testing.pytest import (raises, XFAIL, slow, skip, skip_under_pyodide,
from sympy.assumptions import Q
from sympy.tensor.array import Array
from sympy.matrices.expressions import MatPow
from sympy.algebras import Quaternion
from sympy.abc import a, b, c, d, x, y, z, t
def test_issue_11238():
    from sympy.geometry.point import Point
    xx = 8 * tan(pi * Rational(13, 45)) / (tan(pi * Rational(13, 45)) + sqrt(3))
    yy = (-8 * sqrt(3) * tan(pi * Rational(13, 45)) ** 2 + 24 * tan(pi * Rational(13, 45))) / (-3 + tan(pi * Rational(13, 45)) ** 2)
    p1 = Point(0, 0)
    p2 = Point(1, -sqrt(3))
    p0 = Point(xx, yy)
    m1 = Matrix([p1 - simplify(p0), p2 - simplify(p0)])
    m2 = Matrix([p1 - p0, p2 - p0])
    m3 = Matrix([simplify(p1 - p0), simplify(p2 - p0)])
    Z = lambda x: abs(x.n()) < 1e-20
    assert m1.rank(simplify=True, iszerofunc=Z) == 1
    assert m2.rank(simplify=True, iszerofunc=Z) == 1
    assert m3.rank(simplify=True, iszerofunc=Z) == 1