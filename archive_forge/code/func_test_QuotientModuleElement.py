from sympy.polys.agca.modules import FreeModule, ModuleOrder, FreeModulePolyRing
from sympy.polys import CoercionFailed, QQ, lex, grlex, ilex, ZZ
from sympy.abc import x, y, z
from sympy.testing.pytest import raises
from sympy.core.numbers import Rational
def test_QuotientModuleElement():
    R = QQ.old_poly_ring(x)
    F = R.free_module(3)
    N = F.submodule([1, x, x ** 2])
    M = F / N
    e = M.convert([x ** 2, 2, 0])
    assert M.convert([x + 1, x ** 2 + x, x ** 3 + x ** 2]) == 0
    assert e == [x ** 2, 2, 0] + N == F.convert([x ** 2, 2, 0]) + N == M.convert(F.convert([x ** 2, 2, 0]))
    assert M.convert([x ** 2 + 1, 2 * x + 2, x ** 2]) == e + [0, x, 0] == e + M.convert([0, x, 0]) == e + F.convert([0, x, 0])
    assert M.convert([x ** 2 + 1, 2, x ** 2]) == e - [0, x, 0] == e - M.convert([0, x, 0]) == e - F.convert([0, x, 0])
    assert M.convert([0, 2, 0]) == M.convert([x ** 2, 4, 0]) - e == [x ** 2, 4, 0] - e == F.convert([x ** 2, 4, 0]) - e
    assert M.convert([x ** 3 + x ** 2, 2 * x + 2, 0]) == (1 + x) * e == R.convert(1 + x) * e == e * (1 + x) == e * R.convert(1 + x)
    assert -e == [-x ** 2, -2, 0]
    f = [x, x, 0] + N
    assert M.convert([1, 1, 0]) == f / x == f / R.convert(x)
    M2 = F / [(2, 2 * x, 2 * x ** 2), (0, 0, 1)]
    G = R.free_module(2)
    M3 = G / [[1, x]]
    M4 = F.submodule([1, x, x ** 2], [1, 0, 0]) / N
    raises(CoercionFailed, lambda: M.convert(G.convert([1, x])))
    raises(CoercionFailed, lambda: M.convert(M3.convert([1, x])))
    raises(CoercionFailed, lambda: M.convert(M2.convert([1, x, x])))
    assert M2.convert(M.convert([2, x, x ** 2])) == [2, x, 0]
    assert M.convert(M4.convert([2, 0, 0])) == [2, 0, 0]