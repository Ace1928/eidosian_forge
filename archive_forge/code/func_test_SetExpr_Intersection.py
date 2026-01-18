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
def test_SetExpr_Intersection():
    x, y, z, w = symbols('x y z w')
    set1 = Interval(x, y)
    set2 = Interval(w, z)
    inter = Intersection(set1, set2)
    se = SetExpr(inter)
    assert exp(se).set == Intersection(ImageSet(Lambda(x, exp(x)), set1), ImageSet(Lambda(x, exp(x)), set2))
    assert cos(se).set == ImageSet(Lambda(x, cos(x)), inter)