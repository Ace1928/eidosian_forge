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
def test_Poly__unify():
    raises(UnificationFailed, lambda: Poly(x)._unify(y))
    F3 = FF(3)
    F5 = FF(5)
    assert Poly(x, x, modulus=3)._unify(Poly(y, y, modulus=3))[2:] == (DMP([[F3(1)], []], F3), DMP([[F3(1), F3(0)]], F3))
    assert Poly(x, x, modulus=3)._unify(Poly(y, y, modulus=5))[2:] == (DMP([[F5(1)], []], F5), DMP([[F5(1), F5(0)]], F5))
    assert Poly(y, x, y)._unify(Poly(x, x, modulus=3))[2:] == (DMP([[F3(1), F3(0)]], F3), DMP([[F3(1)], []], F3))
    assert Poly(x, x, modulus=3)._unify(Poly(y, x, y))[2:] == (DMP([[F3(1)], []], F3), DMP([[F3(1), F3(0)]], F3))
    assert Poly(x + 1, x)._unify(Poly(x + 2, x))[2:] == (DMP([1, 1], ZZ), DMP([1, 2], ZZ))
    assert Poly(x + 1, x, domain='QQ')._unify(Poly(x + 2, x))[2:] == (DMP([1, 1], QQ), DMP([1, 2], QQ))
    assert Poly(x + 1, x)._unify(Poly(x + 2, x, domain='QQ'))[2:] == (DMP([1, 1], QQ), DMP([1, 2], QQ))
    assert Poly(x + 1, x)._unify(Poly(x + 2, x, y))[2:] == (DMP([[1], [1]], ZZ), DMP([[1], [2]], ZZ))
    assert Poly(x + 1, x, domain='QQ')._unify(Poly(x + 2, x, y))[2:] == (DMP([[1], [1]], QQ), DMP([[1], [2]], QQ))
    assert Poly(x + 1, x)._unify(Poly(x + 2, x, y, domain='QQ'))[2:] == (DMP([[1], [1]], QQ), DMP([[1], [2]], QQ))
    assert Poly(x + 1, x, y)._unify(Poly(x + 2, x))[2:] == (DMP([[1], [1]], ZZ), DMP([[1], [2]], ZZ))
    assert Poly(x + 1, x, y, domain='QQ')._unify(Poly(x + 2, x))[2:] == (DMP([[1], [1]], QQ), DMP([[1], [2]], QQ))
    assert Poly(x + 1, x, y)._unify(Poly(x + 2, x, domain='QQ'))[2:] == (DMP([[1], [1]], QQ), DMP([[1], [2]], QQ))
    assert Poly(x + 1, x, y)._unify(Poly(x + 2, x, y))[2:] == (DMP([[1], [1]], ZZ), DMP([[1], [2]], ZZ))
    assert Poly(x + 1, x, y, domain='QQ')._unify(Poly(x + 2, x, y))[2:] == (DMP([[1], [1]], QQ), DMP([[1], [2]], QQ))
    assert Poly(x + 1, x, y)._unify(Poly(x + 2, x, y, domain='QQ'))[2:] == (DMP([[1], [1]], QQ), DMP([[1], [2]], QQ))
    assert Poly(x + 1, x)._unify(Poly(x + 2, y, x))[2:] == (DMP([[1, 1]], ZZ), DMP([[1, 2]], ZZ))
    assert Poly(x + 1, x, domain='QQ')._unify(Poly(x + 2, y, x))[2:] == (DMP([[1, 1]], QQ), DMP([[1, 2]], QQ))
    assert Poly(x + 1, x)._unify(Poly(x + 2, y, x, domain='QQ'))[2:] == (DMP([[1, 1]], QQ), DMP([[1, 2]], QQ))
    assert Poly(x + 1, y, x)._unify(Poly(x + 2, x))[2:] == (DMP([[1, 1]], ZZ), DMP([[1, 2]], ZZ))
    assert Poly(x + 1, y, x, domain='QQ')._unify(Poly(x + 2, x))[2:] == (DMP([[1, 1]], QQ), DMP([[1, 2]], QQ))
    assert Poly(x + 1, y, x)._unify(Poly(x + 2, x, domain='QQ'))[2:] == (DMP([[1, 1]], QQ), DMP([[1, 2]], QQ))
    assert Poly(x + 1, x, y)._unify(Poly(x + 2, y, x))[2:] == (DMP([[1], [1]], ZZ), DMP([[1], [2]], ZZ))
    assert Poly(x + 1, x, y, domain='QQ')._unify(Poly(x + 2, y, x))[2:] == (DMP([[1], [1]], QQ), DMP([[1], [2]], QQ))
    assert Poly(x + 1, x, y)._unify(Poly(x + 2, y, x, domain='QQ'))[2:] == (DMP([[1], [1]], QQ), DMP([[1], [2]], QQ))
    assert Poly(x + 1, y, x)._unify(Poly(x + 2, x, y))[2:] == (DMP([[1, 1]], ZZ), DMP([[1, 2]], ZZ))
    assert Poly(x + 1, y, x, domain='QQ')._unify(Poly(x + 2, x, y))[2:] == (DMP([[1, 1]], QQ), DMP([[1, 2]], QQ))
    assert Poly(x + 1, y, x)._unify(Poly(x + 2, x, y, domain='QQ'))[2:] == (DMP([[1, 1]], QQ), DMP([[1, 2]], QQ))
    assert Poly(x ** 2 + I, x, domain=ZZ_I).unify(Poly(x ** 2 + sqrt(2), x, extension=True)) == (Poly(x ** 2 + I, x, domain='QQ<sqrt(2) + I>'), Poly(x ** 2 + sqrt(2), x, domain='QQ<sqrt(2) + I>'))
    F, A, B = field('a,b', ZZ)
    assert Poly(a * x, x, domain='ZZ[a]')._unify(Poly(a * b * x, x, domain='ZZ(a,b)'))[2:] == (DMP([A, F(0)], F.to_domain()), DMP([A * B, F(0)], F.to_domain()))
    assert Poly(a * x, x, domain='ZZ(a)')._unify(Poly(a * b * x, x, domain='ZZ(a,b)'))[2:] == (DMP([A, F(0)], F.to_domain()), DMP([A * B, F(0)], F.to_domain()))
    raises(CoercionFailed, lambda: Poly(Poly(x ** 2 + x ** 2 * z, y, field=True), domain='ZZ(x)'))
    f = Poly(t ** 2 + t / 3 + x, t, domain='QQ(x)')
    g = Poly(t ** 2 + t / 3 + x, t, domain='QQ[x]')
    assert f._unify(g)[2:] == (f.rep, f.rep)