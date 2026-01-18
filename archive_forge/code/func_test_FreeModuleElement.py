from sympy.polys.agca.modules import FreeModule, ModuleOrder, FreeModulePolyRing
from sympy.polys import CoercionFailed, QQ, lex, grlex, ilex, ZZ
from sympy.abc import x, y, z
from sympy.testing.pytest import raises
from sympy.core.numbers import Rational
def test_FreeModuleElement():
    M = QQ.old_poly_ring(x).free_module(3)
    e = M.convert([1, x, x ** 2])
    f = [QQ.old_poly_ring(x).convert(1), QQ.old_poly_ring(x).convert(x), QQ.old_poly_ring(x).convert(x ** 2)]
    assert list(e) == f
    assert f[0] == e[0]
    assert f[1] == e[1]
    assert f[2] == e[2]
    raises(IndexError, lambda: e[3])
    g = M.convert([x, 0, 0])
    assert e + g == M.convert([x + 1, x, x ** 2])
    assert f + g == M.convert([x + 1, x, x ** 2])
    assert -e == M.convert([-1, -x, -x ** 2])
    assert e - g == M.convert([1 - x, x, x ** 2])
    assert e != g
    assert M.convert([x, x, x]) / QQ.old_poly_ring(x).convert(x) == [1, 1, 1]
    R = QQ.old_poly_ring(x, order='ilex')
    assert R.free_module(1).convert([x]) / R.convert(x) == [1]