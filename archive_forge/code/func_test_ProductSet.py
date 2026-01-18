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
def test_ProductSet():
    assert ProductSet(S.Reals) == S.Reals ** 1
    assert ProductSet(S.Reals, S.Reals) == S.Reals ** 2
    assert ProductSet(S.Reals, S.Reals, S.Reals) == S.Reals ** 3
    assert ProductSet(S.Reals) != S.Reals
    assert ProductSet(S.Reals, S.Reals) == S.Reals * S.Reals
    assert ProductSet(S.Reals, S.Reals, S.Reals) != S.Reals * S.Reals * S.Reals
    assert ProductSet(S.Reals, S.Reals, S.Reals) == (S.Reals * S.Reals * S.Reals).flatten()
    assert 1 not in ProductSet(S.Reals)
    assert (1,) in ProductSet(S.Reals)
    assert 1 not in ProductSet(S.Reals, S.Reals)
    assert (1, 2) in ProductSet(S.Reals, S.Reals)
    assert (1, I) not in ProductSet(S.Reals, S.Reals)
    assert (1, 2, 3) in ProductSet(S.Reals, S.Reals, S.Reals)
    assert (1, 2, 3) in S.Reals ** 3
    assert (1, 2, 3) not in S.Reals * S.Reals * S.Reals
    assert ((1, 2), 3) in S.Reals * S.Reals * S.Reals
    assert (1, (2, 3)) not in S.Reals * S.Reals * S.Reals
    assert (1, (2, 3)) in S.Reals * (S.Reals * S.Reals)
    assert ProductSet() == FiniteSet(())
    assert ProductSet(S.Reals, S.EmptySet) == S.EmptySet
    for ni in range(5):
        Rn = ProductSet(*(S.Reals,) * ni)
        assert (1,) * ni in Rn
        assert 1 not in Rn
    assert S.Reals * S.Reals * S.Reals != S.Reals * (S.Reals * S.Reals)
    S1 = S.Reals
    S2 = S.Integers
    x1 = pi
    x2 = 3
    assert x1 in S1
    assert x2 in S2
    assert (x1, x2) in S1 * S2
    S3 = S1 * S2
    x3 = (x1, x2)
    assert x3 in S3
    assert (x3, x3) in S3 * S3
    assert x3 + x3 not in S3 * S3
    raises(ValueError, lambda: S.Reals ** (-1))
    with warns_deprecated_sympy():
        ProductSet((FiniteSet(s) for s in range(2)))
    raises(TypeError, lambda: ProductSet(None))
    S1 = FiniteSet(1, 2)
    S2 = FiniteSet(3, 4)
    S3 = ProductSet(S1, S2)
    assert S3.as_relational(x, y) == And(S1.as_relational(x), S2.as_relational(y)) == And(Or(Eq(x, 1), Eq(x, 2)), Or(Eq(y, 3), Eq(y, 4)))
    raises(ValueError, lambda: S3.as_relational(x))
    raises(ValueError, lambda: S3.as_relational(x, 1))
    raises(ValueError, lambda: ProductSet(Interval(0, 1)).as_relational(x, y))
    Z2 = ProductSet(S.Integers, S.Integers)
    assert Z2.contains((1, 2)) is S.true
    assert Z2.contains((1,)) is S.false
    assert Z2.contains(x) == Contains(x, Z2, evaluate=False)
    assert Z2.contains(x).subs(x, 1) is S.false
    assert Z2.contains((x, 1)).subs(x, 2) is S.true
    assert Z2.contains((x, y)) == Contains((x, y), Z2, evaluate=False)
    assert unchanged(Contains, (x, y), Z2)
    assert Contains((1, 2), Z2) is S.true