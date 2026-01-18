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
def test_SymmetricDifference():
    A = FiniteSet(0, 1, 2, 3, 4, 5)
    B = FiniteSet(2, 4, 6, 8, 10)
    C = Interval(8, 10)
    assert SymmetricDifference(A, B, evaluate=False).is_iterable is True
    assert SymmetricDifference(A, C, evaluate=False).is_iterable is None
    assert FiniteSet(*SymmetricDifference(A, B, evaluate=False)) == FiniteSet(0, 1, 3, 5, 6, 8, 10)
    raises(TypeError, lambda: FiniteSet(*SymmetricDifference(A, C, evaluate=False)))
    assert SymmetricDifference(FiniteSet(0, 1, 2, 3, 4, 5), FiniteSet(2, 4, 6, 8, 10)) == FiniteSet(0, 1, 3, 5, 6, 8, 10)
    assert SymmetricDifference(FiniteSet(2, 3, 4), FiniteSet(2, 3, 4, 5)) == FiniteSet(5)
    assert FiniteSet(1, 2, 3, 4, 5) ^ FiniteSet(1, 2, 5, 6) == FiniteSet(3, 4, 6)
    assert Set(S(1), S(2), S(3)) ^ Set(S(2), S(3), S(4)) == Union(Set(S(1), S(2), S(3)) - Set(S(2), S(3), S(4)), Set(S(2), S(3), S(4)) - Set(S(1), S(2), S(3)))
    assert Interval(0, 4) ^ Interval(2, 5) == Union(Interval(0, 4) - Interval(2, 5), Interval(2, 5) - Interval(0, 4))