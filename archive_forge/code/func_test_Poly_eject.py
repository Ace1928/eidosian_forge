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
def test_Poly_eject():
    f = Poly(x ** 2 * y + x * y ** 3 + x * y + 1, x, y)
    assert f.eject(x) == Poly(x * y ** 3 + (x ** 2 + x) * y + 1, y, domain='ZZ[x]')
    assert f.eject(y) == Poly(y * x ** 2 + (y ** 3 + y) * x + 1, x, domain='ZZ[y]')
    ex = x + y + z + t + w
    g = Poly(ex, x, y, z, t, w)
    assert g.eject(x) == Poly(ex, y, z, t, w, domain='ZZ[x]')
    assert g.eject(x, y) == Poly(ex, z, t, w, domain='ZZ[x, y]')
    assert g.eject(x, y, z) == Poly(ex, t, w, domain='ZZ[x, y, z]')
    assert g.eject(w) == Poly(ex, x, y, z, t, domain='ZZ[w]')
    assert g.eject(t, w) == Poly(ex, x, y, z, domain='ZZ[t, w]')
    assert g.eject(z, t, w) == Poly(ex, x, y, domain='ZZ[z, t, w]')
    raises(DomainError, lambda: Poly(x * y, x, y, domain=ZZ[z]).eject(y))
    raises(NotImplementedError, lambda: Poly(x * y, x, y, z).eject(y))