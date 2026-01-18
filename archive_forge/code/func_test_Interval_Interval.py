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
def test_Interval_Interval():
    assert (SetExpr(Interval(1, 2)) + SetExpr(Interval(10, 20))).set == Interval(11, 22)
    assert (SetExpr(Interval(1, 2)) * SetExpr(Interval(10, 20))).set == Interval(10, 40)