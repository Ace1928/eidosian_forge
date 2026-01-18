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
def test_complement():
    assert Complement({1, 2}, {1}) == {2}
    assert Interval(0, 1).complement(S.Reals) == Union(Interval(-oo, 0, True, True), Interval(1, oo, True, True))
    assert Interval(0, 1, True, False).complement(S.Reals) == Union(Interval(-oo, 0, True, False), Interval(1, oo, True, True))
    assert Interval(0, 1, False, True).complement(S.Reals) == Union(Interval(-oo, 0, True, True), Interval(1, oo, False, True))
    assert Interval(0, 1, True, True).complement(S.Reals) == Union(Interval(-oo, 0, True, False), Interval(1, oo, False, True))
    assert S.UniversalSet.complement(S.EmptySet) == S.EmptySet
    assert S.UniversalSet.complement(S.Reals) == S.EmptySet
    assert S.UniversalSet.complement(S.UniversalSet) == S.EmptySet
    assert S.EmptySet.complement(S.Reals) == S.Reals
    assert Union(Interval(0, 1), Interval(2, 3)).complement(S.Reals) == Union(Interval(-oo, 0, True, True), Interval(1, 2, True, True), Interval(3, oo, True, True))
    assert FiniteSet(0).complement(S.Reals) == Union(Interval(-oo, 0, True, True), Interval(0, oo, True, True))
    assert (FiniteSet(5) + Interval(S.NegativeInfinity, 0)).complement(S.Reals) == Interval(0, 5, True, True) + Interval(5, S.Infinity, True, True)
    assert FiniteSet(1, 2, 3).complement(S.Reals) == Interval(S.NegativeInfinity, 1, True, True) + Interval(1, 2, True, True) + Interval(2, 3, True, True) + Interval(3, S.Infinity, True, True)
    assert FiniteSet(x).complement(S.Reals) == Complement(S.Reals, FiniteSet(x))
    assert FiniteSet(0, x).complement(S.Reals) == Complement(Interval(-oo, 0, True, True) + Interval(0, oo, True, True), FiniteSet(x), evaluate=False)
    square = Interval(0, 1) * Interval(0, 1)
    notsquare = square.complement(S.Reals * S.Reals)
    assert all((pt in square for pt in [(0, 0), (0.5, 0.5), (1, 0), (1, 1)]))
    assert not any((pt in notsquare for pt in [(0, 0), (0.5, 0.5), (1, 0), (1, 1)]))
    assert not any((pt in square for pt in [(-1, 0), (1.5, 0.5), (10, 10)]))
    assert all((pt in notsquare for pt in [(-1, 0), (1.5, 0.5), (10, 10)]))