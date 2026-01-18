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
def test_Interval_as_relational():
    x = Symbol('x')
    assert Interval(-1, 2, False, False).as_relational(x) == And(Le(-1, x), Le(x, 2))
    assert Interval(-1, 2, True, False).as_relational(x) == And(Lt(-1, x), Le(x, 2))
    assert Interval(-1, 2, False, True).as_relational(x) == And(Le(-1, x), Lt(x, 2))
    assert Interval(-1, 2, True, True).as_relational(x) == And(Lt(-1, x), Lt(x, 2))
    assert Interval(-oo, 2, right_open=False).as_relational(x) == And(Lt(-oo, x), Le(x, 2))
    assert Interval(-oo, 2, right_open=True).as_relational(x) == And(Lt(-oo, x), Lt(x, 2))
    assert Interval(-2, oo, left_open=False).as_relational(x) == And(Le(-2, x), Lt(x, oo))
    assert Interval(-2, oo, left_open=True).as_relational(x) == And(Lt(-2, x), Lt(x, oo))
    assert Interval(-oo, oo).as_relational(x) == And(Lt(-oo, x), Lt(x, oo))
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    assert Interval(x, y).as_relational(x) == (x <= y)
    assert Interval(y, x).as_relational(x) == (y <= x)