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
def test_evalf_with_zoo():
    assert (1 / x).evalf(subs={x: 0}) == zoo
    assert (-1 / x).evalf(subs={x: 0}) == zoo
    assert (0 ** x).evalf(subs={x: -1}) == zoo
    assert (0 ** x).evalf(subs={x: -1 + I}) == nan
    assert Mul(2, Pow(0, -1, evaluate=False), evaluate=False).evalf() == zoo
    assert Mul(x, 1 / x, evaluate=False).evalf(subs={x: 0}) == Mul(x, 1 / x, evaluate=False).subs(x, 0) == nan
    assert Mul(1 / x, 1 / x, evaluate=False).evalf(subs={x: 0}) == zoo
    assert Mul(1 / x, Abs(1 / x), evaluate=False).evalf(subs={x: 0}) == zoo
    assert Abs(zoo, evaluate=False).evalf() == oo
    assert re(zoo, evaluate=False).evalf() == nan
    assert im(zoo, evaluate=False).evalf() == nan
    assert Add(zoo, zoo, evaluate=False).evalf() == nan
    assert Add(oo, zoo, evaluate=False).evalf() == nan
    assert Pow(zoo, -1, evaluate=False).evalf() == 0
    assert Pow(zoo, Rational(-1, 3), evaluate=False).evalf() == 0
    assert Pow(zoo, Rational(1, 3), evaluate=False).evalf() == zoo
    assert Pow(zoo, S.Half, evaluate=False).evalf() == zoo
    assert Pow(zoo, 2, evaluate=False).evalf() == zoo
    assert Pow(0, zoo, evaluate=False).evalf() == nan
    assert log(zoo, evaluate=False).evalf() == zoo
    assert zoo.evalf(chop=True) == zoo
    assert x.evalf(subs={x: zoo}) == zoo