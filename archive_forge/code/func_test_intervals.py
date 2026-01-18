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
def test_intervals():
    assert intervals(0) == []
    assert intervals(1) == []
    assert intervals(x, sqf=True) == [(0, 0)]
    assert intervals(x) == [((0, 0), 1)]
    assert intervals(x ** 128) == [((0, 0), 128)]
    assert intervals([x ** 2, x ** 4]) == [((0, 0), {0: 2, 1: 4})]
    f = Poly((x * Rational(2, 5) - Rational(17, 3)) * (4 * x + Rational(1, 257)))
    assert f.intervals(sqf=True) == [(-1, 0), (14, 15)]
    assert f.intervals() == [((-1, 0), 1), ((14, 15), 1)]
    assert f.intervals(fast=True, sqf=True) == [(-1, 0), (14, 15)]
    assert f.intervals(fast=True) == [((-1, 0), 1), ((14, 15), 1)]
    assert f.intervals(eps=Rational(1, 10)) == f.intervals(eps=0.1) == [((Rational(-1, 258), 0), 1), ((Rational(85, 6), Rational(85, 6)), 1)]
    assert f.intervals(eps=Rational(1, 100)) == f.intervals(eps=0.01) == [((Rational(-1, 258), 0), 1), ((Rational(85, 6), Rational(85, 6)), 1)]
    assert f.intervals(eps=Rational(1, 1000)) == f.intervals(eps=0.001) == [((Rational(-1, 1002), 0), 1), ((Rational(85, 6), Rational(85, 6)), 1)]
    assert f.intervals(eps=Rational(1, 10000)) == f.intervals(eps=0.0001) == [((Rational(-1, 1028), Rational(-1, 1028)), 1), ((Rational(85, 6), Rational(85, 6)), 1)]
    f = (x * Rational(2, 5) - Rational(17, 3)) * (4 * x + Rational(1, 257))
    assert intervals(f, sqf=True) == [(-1, 0), (14, 15)]
    assert intervals(f) == [((-1, 0), 1), ((14, 15), 1)]
    assert intervals(f, eps=Rational(1, 10)) == intervals(f, eps=0.1) == [((Rational(-1, 258), 0), 1), ((Rational(85, 6), Rational(85, 6)), 1)]
    assert intervals(f, eps=Rational(1, 100)) == intervals(f, eps=0.01) == [((Rational(-1, 258), 0), 1), ((Rational(85, 6), Rational(85, 6)), 1)]
    assert intervals(f, eps=Rational(1, 1000)) == intervals(f, eps=0.001) == [((Rational(-1, 1002), 0), 1), ((Rational(85, 6), Rational(85, 6)), 1)]
    assert intervals(f, eps=Rational(1, 10000)) == intervals(f, eps=0.0001) == [((Rational(-1, 1028), Rational(-1, 1028)), 1), ((Rational(85, 6), Rational(85, 6)), 1)]
    f = Poly((x ** 2 - 2) * (x ** 2 - 3) ** 7 * (x + 1) * (7 * x + 3) ** 3)
    assert f.intervals() == [((-2, Rational(-3, 2)), 7), ((Rational(-3, 2), -1), 1), ((-1, -1), 1), ((-1, 0), 3), ((1, Rational(3, 2)), 1), ((Rational(3, 2), 2), 7)]
    assert intervals([x ** 5 - 200, x ** 5 - 201]) == [((Rational(75, 26), Rational(101, 35)), {0: 1}), ((Rational(309, 107), Rational(26, 9)), {1: 1})]
    assert intervals([x ** 5 - 200, x ** 5 - 201], fast=True) == [((Rational(75, 26), Rational(101, 35)), {0: 1}), ((Rational(309, 107), Rational(26, 9)), {1: 1})]
    assert intervals([x ** 2 - 200, x ** 2 - 201]) == [((Rational(-71, 5), Rational(-85, 6)), {1: 1}), ((Rational(-85, 6), -14), {0: 1}), ((14, Rational(85, 6)), {0: 1}), ((Rational(85, 6), Rational(71, 5)), {1: 1})]
    assert intervals([x + 1, x + 2, x - 1, x + 1, 1, x - 1, x - 1, (x - 2) ** 2]) == [((-2, -2), {1: 1}), ((-1, -1), {0: 1, 3: 1}), ((1, 1), {2: 1, 5: 1, 6: 1}), ((2, 2), {7: 2})]
    f, g, h = (x ** 2 - 2, x ** 4 - 4 * x ** 2 + 4, x - 1)
    assert intervals(f, inf=Rational(7, 4), sqf=True) == []
    assert intervals(f, inf=Rational(7, 5), sqf=True) == [(Rational(7, 5), Rational(3, 2))]
    assert intervals(f, sup=Rational(7, 4), sqf=True) == [(-2, -1), (1, Rational(3, 2))]
    assert intervals(f, sup=Rational(7, 5), sqf=True) == [(-2, -1)]
    assert intervals(g, inf=Rational(7, 4)) == []
    assert intervals(g, inf=Rational(7, 5)) == [((Rational(7, 5), Rational(3, 2)), 2)]
    assert intervals(g, sup=Rational(7, 4)) == [((-2, -1), 2), ((1, Rational(3, 2)), 2)]
    assert intervals(g, sup=Rational(7, 5)) == [((-2, -1), 2)]
    assert intervals([g, h], inf=Rational(7, 4)) == []
    assert intervals([g, h], inf=Rational(7, 5)) == [((Rational(7, 5), Rational(3, 2)), {0: 2})]
    assert intervals([g, h], sup=S(7) / 4) == [((-2, -1), {0: 2}), ((1, 1), {1: 1}), ((1, Rational(3, 2)), {0: 2})]
    assert intervals([g, h], sup=Rational(7, 5)) == [((-2, -1), {0: 2}), ((1, 1), {1: 1})]
    assert intervals([x + 2, x ** 2 - 2]) == [((-2, -2), {0: 1}), ((-2, -1), {1: 1}), ((1, 2), {1: 1})]
    assert intervals([x + 2, x ** 2 - 2], strict=True) == [((-2, -2), {0: 1}), ((Rational(-3, 2), -1), {1: 1}), ((1, 2), {1: 1})]
    f = 7 * z ** 4 - 19 * z ** 3 + 20 * z ** 2 + 17 * z + 20
    assert intervals(f) == []
    real_part, complex_part = intervals(f, all=True, sqf=True)
    assert real_part == []
    assert all((re(a) < re(r) < re(b) and im(a) < im(r) < im(b) for (a, b), r in zip(complex_part, nroots(f))))
    assert complex_part == [(Rational(-40, 7) - I * 40 / 7, 0), (Rational(-40, 7), I * 40 / 7), (I * Rational(-40, 7), Rational(40, 7)), (0, Rational(40, 7) + I * 40 / 7)]
    real_part, complex_part = intervals(f, all=True, sqf=True, eps=Rational(1, 10))
    assert real_part == []
    assert all((re(a) < re(r) < re(b) and im(a) < im(r) < im(b) for (a, b), r in zip(complex_part, nroots(f))))
    raises(ValueError, lambda: intervals(x ** 2 - 2, eps=10 ** (-100000)))
    raises(ValueError, lambda: Poly(x ** 2 - 2).intervals(eps=10 ** (-100000)))
    raises(ValueError, lambda: intervals([x ** 2 - 2, x ** 2 - 3], eps=10 ** (-100000)))