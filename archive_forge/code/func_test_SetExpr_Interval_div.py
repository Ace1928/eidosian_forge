from sympy.sets.setexpr import SetExpr
from sympy.sets import Interval, FiniteSet, Intersection, ImageSet, Union
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.numbers import (I, Rational, oo)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.trigonometric import cos
from sympy.sets.sets import Set
def test_SetExpr_Interval_div():
    assert SetExpr(Interval(-3, -2)) / SetExpr(Interval(-2, 1)) == SetExpr(Interval(-oo, oo))
    assert SetExpr(Interval(2, 3)) / SetExpr(Interval(-2, 2)) == SetExpr(Interval(-oo, oo))
    assert SetExpr(Interval(-3, -2)) / SetExpr(Interval(0, 4)) == SetExpr(Interval(-oo, Rational(-1, 2)))
    assert SetExpr(Interval(2, 4)) / SetExpr(Interval(-3, 0)) == SetExpr(Interval(-oo, Rational(-2, 3)))
    assert SetExpr(Interval(2, 4)) / SetExpr(Interval(0, 3)) == SetExpr(Interval(Rational(2, 3), oo))
    assert SetExpr(Interval(-1, 2)) / SetExpr(Interval(-2, 2)) == SetExpr(Interval(-oo, oo))
    assert 1 / SetExpr(Interval(-1, 2)) == SetExpr(Union(Interval(-oo, -1), Interval(S.Half, oo)))
    assert 1 / SetExpr(Interval(0, 2)) == SetExpr(Interval(S.Half, oo))
    assert -1 / SetExpr(Interval(0, 2)) == SetExpr(Interval(-oo, Rational(-1, 2)))
    assert 1 / SetExpr(Interval(-oo, 0)) == SetExpr(Interval.open(-oo, 0))
    assert 1 / SetExpr(Interval(-1, 0)) == SetExpr(Interval(-oo, -1))