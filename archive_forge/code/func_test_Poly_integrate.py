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
def test_Poly_integrate():
    assert Poly(x + 1).integrate() == Poly(x ** 2 / 2 + x)
    assert Poly(x + 1).integrate(x) == Poly(x ** 2 / 2 + x)
    assert Poly(x + 1).integrate((x, 1)) == Poly(x ** 2 / 2 + x)
    assert Poly(x * y + 1).integrate(x) == Poly(x ** 2 * y / 2 + x)
    assert Poly(x * y + 1).integrate(y) == Poly(x * y ** 2 / 2 + y)
    assert Poly(x * y + 1).integrate(x, x) == Poly(x ** 3 * y / 6 + x ** 2 / 2)
    assert Poly(x * y + 1).integrate(y, y) == Poly(x * y ** 3 / 6 + y ** 2 / 2)
    assert Poly(x * y + 1).integrate((x, 2)) == Poly(x ** 3 * y / 6 + x ** 2 / 2)
    assert Poly(x * y + 1).integrate((y, 2)) == Poly(x * y ** 3 / 6 + y ** 2 / 2)
    assert Poly(x * y + 1).integrate(x, y) == Poly(x ** 2 * y ** 2 / 4 + x * y)
    assert Poly(x * y + 1).integrate(y, x) == Poly(x ** 2 * y ** 2 / 4 + x * y)