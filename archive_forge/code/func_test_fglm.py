import pickle
from sympy.polys.polytools import (
from sympy.polys.polyerrors import (
from sympy.polys.polyclasses import DMP
from sympy.polys.fields import field
from sympy.polys.domains import FF, ZZ, QQ, ZZ_I, QQ_I, RR, EX
from sympy.polys.domains.realfield import RealField
from sympy.polys.domains.complexfield import ComplexField
from sympy.polys.orderings import lex, grlex, grevlex
from sympy.combinatorics.galois import S4TransitiveSubgroups
from sympy.core.add import Add
from sympy.core.basic import _aresame
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import (Derivative, diff, expand)
from sympy.core.mul import _keep_coeff, Mul
from sympy.core.numbers import (Float, I, Integer, Rational, oo, pi)
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import sin
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.polys.rootoftools import rootof
from sympy.simplify.simplify import signsimp
from sympy.utilities.iterables import iterable
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.testing.pytest import raises, warns_deprecated_sympy, warns
from sympy.abc import a, b, c, d, p, q, t, w, x, y, z
def test_fglm():
    F = [a + b + c + d, a * b + a * d + b * c + b * d, a * b * c + a * b * d + a * c * d + b * c * d, a * b * c * d - 1]
    G = groebner(F, a, b, c, d, order=grlex)
    B = [4 * a + 3 * d ** 9 - 4 * d ** 5 - 3 * d, 4 * b + 4 * c - 3 * d ** 9 + 4 * d ** 5 + 7 * d, 4 * c ** 2 + 3 * d ** 10 - 4 * d ** 6 - 3 * d ** 2, 4 * c * d ** 4 + 4 * c - d ** 9 + 4 * d ** 5 + 5 * d, d ** 12 - d ** 8 - d ** 4 + 1]
    assert groebner(F, a, b, c, d, order=lex) == B
    assert G.fglm(lex) == B
    F = [9 * x ** 8 + 36 * x ** 7 - 32 * x ** 6 - 252 * x ** 5 - 78 * x ** 4 + 468 * x ** 3 + 288 * x ** 2 - 108 * x + 9, -72 * t * x ** 7 - 252 * t * x ** 6 + 192 * t * x ** 5 + 1260 * t * x ** 4 + 312 * t * x ** 3 - 404 * t * x ** 2 - 576 * t * x + 108 * t - 72 * x ** 7 - 256 * x ** 6 + 192 * x ** 5 + 1280 * x ** 4 + 312 * x ** 3 - 576 * x + 96]
    G = groebner(F, t, x, order=grlex)
    B = [203577793572507451707 * t + 627982239411707112 * x ** 7 - 666924143779443762 * x ** 6 - 10874593056632447619 * x ** 5 + 5119998792707079562 * x ** 4 + 72917161949456066376 * x ** 3 + 20362663855832380362 * x ** 2 - 142079311455258371571 * x + 183756699868981873194, 9 * x ** 8 + 36 * x ** 7 - 32 * x ** 6 - 252 * x ** 5 - 78 * x ** 4 + 468 * x ** 3 + 288 * x ** 2 - 108 * x + 9]
    assert groebner(F, t, x, order=lex) == B
    assert G.fglm(lex) == B
    F = [x ** 2 - x - 3 * y + 1, -2 * x + y ** 2 + y - 1]
    G = groebner(F, x, y, order=lex)
    B = [x ** 2 - x - 3 * y + 1, y ** 2 - 2 * x + y - 1]
    assert groebner(F, x, y, order=grlex) == B
    assert G.fglm(grlex) == B