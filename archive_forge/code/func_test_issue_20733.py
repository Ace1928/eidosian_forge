import math
from sympy.concrete.products import (Product, product)
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.evalf import N
from sympy.core.function import (Function, nfloat)
from sympy.core.mul import Mul
from sympy.core import (GoldenRatio)
from sympy.core.numbers import (AlgebraicNumber, E, Float, I, Rational,
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.combinatorial.numbers import fibonacci
from sympy.functions.elementary.complexes import (Abs, re, im)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, cosh)
from sympy.functions.elementary.integers import (ceiling, floor)
from sympy.functions.elementary.miscellaneous import (Max, sqrt)
from sympy.functions.elementary.trigonometric import (acos, atan, cos, sin, tan)
from sympy.integrals.integrals import (Integral, integrate)
from sympy.polys.polytools import factor
from sympy.polys.rootoftools import CRootOf
from sympy.polys.specialpolys import cyclotomic_poly
from sympy.printing import srepr
from sympy.printing.str import sstr
from sympy.simplify.simplify import simplify
from sympy.core.numbers import comp
from sympy.core.evalf import (complex_accuracy, PrecisionExhausted,
from mpmath import inf, ninf, make_mpc
from mpmath.libmp.libmpf import from_float, fzero
from sympy.core.expr import unchanged
from sympy.testing.pytest import raises, XFAIL
from sympy.abc import n, x, y
def test_issue_20733():
    expr = 1 / ((x - 9) * (x - 8) * (x - 7) * (x - 4) ** 2 * (x - 3) ** 3 * (x - 2))
    assert str(expr.evalf(1, subs={x: 1})) == '-4.e-5'
    assert str(expr.evalf(2, subs={x: 1})) == '-4.1e-5'
    assert str(expr.evalf(11, subs={x: 1})) == '-4.1335978836e-5'
    assert str(expr.evalf(20, subs={x: 1})) == '-0.000041335978835978835979'
    expr = Mul(*(x - i for i in range(2, 1000)))
    assert srepr(expr.evalf(2, subs={x: 1})) == "Float('4.0271e+2561', precision=10)"
    assert srepr(expr.evalf(10, subs={x: 1})) == "Float('4.02790050126e+2561', precision=37)"
    assert srepr(expr.evalf(53, subs={x: 1})) == "Float('4.0279005012722099453824067459760158730668154575647110393e+2561', precision=179)"