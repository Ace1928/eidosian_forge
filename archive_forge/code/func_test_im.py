from sympy.assumptions.ask import Q
from sympy.assumptions.refine import refine
from sympy.core.expr import Expr
from sympy.core.numbers import (I, Rational, nan, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import (Abs, arg, im, re, sign)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (atan, atan2)
from sympy.abc import w, x, y, z
from sympy.core.relational import Eq, Ne
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices.expressions.matexpr import MatrixSymbol
def test_im():
    assert refine(im(x), Q.imaginary(x)) == -I * x
    assert refine(im(x), Q.real(x)) is S.Zero
    assert refine(im(x + y), Q.imaginary(x) & Q.imaginary(y)) == -I * x - I * y
    assert refine(im(x + y), Q.real(x) & Q.imaginary(y)) == -I * y
    assert refine(im(x * y), Q.imaginary(x) & Q.real(y)) == -I * x * y
    assert refine(im(x * y), Q.imaginary(x) & Q.imaginary(y)) == 0
    assert refine(im(1 / x), Q.imaginary(x)) == -I / x
    assert refine(im(x * y * z), Q.imaginary(x) & Q.imaginary(y) & Q.imaginary(z)) == -I * x * y * z