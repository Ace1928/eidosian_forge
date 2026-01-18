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
def test_issue_4956_5204():
    v = S('(-27*12**(1/3)*sqrt(31)*I +\n    27*2**(2/3)*3**(1/3)*sqrt(31)*I)/(-2511*2**(2/3)*3**(1/3) +\n    (29*18**(1/3) + 9*2**(1/3)*3**(2/3)*sqrt(31)*I +\n    87*2**(1/3)*3**(1/6)*I)**2)')
    assert NS(v, 1) == '0.e-118 - 0.e-118*I'
    v = S('-(357587765856 + 18873261792*249**(1/2) + 56619785376*I*83**(1/2) +\n    108755765856*I*3**(1/2) + 41281887168*6**(1/3)*(1422 +\n    54*249**(1/2))**(1/3) - 1239810624*6**(1/3)*249**(1/2)*(1422 +\n    54*249**(1/2))**(1/3) - 3110400000*I*6**(1/3)*83**(1/2)*(1422 +\n    54*249**(1/2))**(1/3) + 13478400000*I*3**(1/2)*6**(1/3)*(1422 +\n    54*249**(1/2))**(1/3) + 1274950152*6**(2/3)*(1422 +\n    54*249**(1/2))**(2/3) + 32347944*6**(2/3)*249**(1/2)*(1422 +\n    54*249**(1/2))**(2/3) - 1758790152*I*3**(1/2)*6**(2/3)*(1422 +\n    54*249**(1/2))**(2/3) - 304403832*I*6**(2/3)*83**(1/2)*(1422 +\n    4*249**(1/2))**(2/3))/(175732658352 + (1106028 + 25596*249**(1/2) +\n    76788*I*83**(1/2))**2)')
    assert NS(v, 5) == '0.077284 + 1.1104*I'
    assert NS(v, 1) == '0.08 + 1.*I'