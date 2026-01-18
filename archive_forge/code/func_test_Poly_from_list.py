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
def test_Poly_from_list():
    K = FF(3)
    assert Poly.from_list([2, 1], gens=x, domain=K).rep == DMP([K(2), K(1)], K)
    assert Poly.from_list([5, 1], gens=x, domain=K).rep == DMP([K(2), K(1)], K)
    assert Poly.from_list([2, 1], gens=x).rep == DMP([ZZ(2), ZZ(1)], ZZ)
    assert Poly.from_list([2, 1], gens=x, field=True).rep == DMP([QQ(2), QQ(1)], QQ)
    assert Poly.from_list([2, 1], gens=x, domain=ZZ).rep == DMP([ZZ(2), ZZ(1)], ZZ)
    assert Poly.from_list([2, 1], gens=x, domain=QQ).rep == DMP([QQ(2), QQ(1)], QQ)
    assert Poly.from_list([0, 1.0], gens=x).rep == DMP([RR(1.0)], RR)
    assert Poly.from_list([1.0, 0], gens=x).rep == DMP([RR(1.0), RR(0.0)], RR)
    raises(MultivariatePolynomialError, lambda: Poly.from_list([[]], gens=(x, y)))