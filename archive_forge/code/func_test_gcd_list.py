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
def test_gcd_list():
    F = [x ** 3 - 1, x ** 2 - 1, x ** 2 - 3 * x + 2]
    assert gcd_list(F) == x - 1
    assert gcd_list(F, polys=True) == Poly(x - 1)
    assert gcd_list([]) == 0
    assert gcd_list([1, 2]) == 1
    assert gcd_list([4, 6, 8]) == 2
    assert gcd_list([x * (y + 42) - x * y - x * 42]) == 0
    gcd = gcd_list([], x)
    assert gcd.is_Number and gcd is S.Zero
    gcd = gcd_list([], x, polys=True)
    assert gcd.is_Poly and gcd.is_zero
    a = sqrt(2)
    assert gcd_list([a, -a]) == gcd_list([-a, a]) == a
    raises(ComputationFailed, lambda: gcd_list([], polys=True))