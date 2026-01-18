from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.containers import TupleKind
from sympy.core.function import Lambda
from sympy.core.kind import NumberKind, UndefinedKind
from sympy.core.numbers import (Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.logic.boolalg import (false, true)
from sympy.matrices.common import MatrixKind
from sympy.matrices.dense import Matrix
from sympy.polys.rootoftools import rootof
from sympy.sets.contains import Contains
from sympy.sets.fancysets import (ImageSet, Range)
from sympy.sets.sets import (Complement, DisjointUnion, FiniteSet, Intersection, Interval, ProductSet, Set, SymmetricDifference, Union, imageset, SetKind)
from mpmath import mpi
from sympy.core.expr import unchanged
from sympy.core.relational import Eq, Ne, Le, Lt, LessThan
from sympy.logic import And, Or, Xor
from sympy.testing.pytest import raises, XFAIL, warns_deprecated_sympy
from sympy.abc import x, y, z, m, n
def test_image_interval():
    x = Symbol('x', real=True)
    a = Symbol('a', real=True)
    assert imageset(x, 2 * x, Interval(-2, 1)) == Interval(-4, 2)
    assert imageset(x, 2 * x, Interval(-2, 1, True, False)) == Interval(-4, 2, True, False)
    assert imageset(x, x ** 2, Interval(-2, 1, True, False)) == Interval(0, 4, False, True)
    assert imageset(x, x ** 2, Interval(-2, 1)) == Interval(0, 4)
    assert imageset(x, x ** 2, Interval(-2, 1, True, False)) == Interval(0, 4, False, True)
    assert imageset(x, x ** 2, Interval(-2, 1, True, True)) == Interval(0, 4, False, True)
    assert imageset(x, (x - 2) ** 2, Interval(1, 3)) == Interval(0, 1)
    assert imageset(x, 3 * x ** 4 - 26 * x ** 3 + 78 * x ** 2 - 90 * x, Interval(0, 4)) == Interval(-35, 0)
    assert imageset(x, x + 1 / x, Interval(-oo, oo)) == Interval(-oo, -2) + Interval(2, oo)
    assert imageset(x, 1 / x + 1 / (x - 1) ** 2, Interval(0, 2, True, False)) == Interval(Rational(3, 2), oo, False)
    assert imageset(lambda x: 2 * x, Interval(-2, 1)) == Interval(-4, 2)
    assert imageset(Lambda(x, a * x), Interval(0, 1)) == ImageSet(Lambda(x, a * x), Interval(0, 1))
    assert imageset(Lambda(x, sin(cos(x))), Interval(0, 1)) == ImageSet(Lambda(x, sin(cos(x))), Interval(0, 1))