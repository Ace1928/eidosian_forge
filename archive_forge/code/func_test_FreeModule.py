from sympy.polys.agca.modules import FreeModule, ModuleOrder, FreeModulePolyRing
from sympy.polys import CoercionFailed, QQ, lex, grlex, ilex, ZZ
from sympy.abc import x, y, z
from sympy.testing.pytest import raises
from sympy.core.numbers import Rational
def test_FreeModule():
    M1 = FreeModule(QQ.old_poly_ring(x), 2)
    assert M1 == FreeModule(QQ.old_poly_ring(x), 2)
    assert M1 != FreeModule(QQ.old_poly_ring(y), 2)
    assert M1 != FreeModule(QQ.old_poly_ring(x), 3)
    M2 = FreeModule(QQ.old_poly_ring(x, order='ilex'), 2)
    assert [x, 1] in M1
    assert [x] not in M1
    assert [2, y] not in M1
    assert [1 / (x + 1), 2] not in M1
    e = M1.convert([x, x ** 2 + 1])
    X = QQ.old_poly_ring(x).convert(x)
    assert e == [X, X ** 2 + 1]
    assert e == [x, x ** 2 + 1]
    assert 2 * e == [2 * x, 2 * x ** 2 + 2]
    assert e * 2 == [2 * x, 2 * x ** 2 + 2]
    assert e / 2 == [x / 2, (x ** 2 + 1) / 2]
    assert x * e == [x ** 2, x ** 3 + x]
    assert e * x == [x ** 2, x ** 3 + x]
    assert X * e == [x ** 2, x ** 3 + x]
    assert e * X == [x ** 2, x ** 3 + x]
    assert [x, 1] in M2
    assert [x] not in M2
    assert [2, y] not in M2
    assert [1 / (x + 1), 2] in M2
    e = M2.convert([x, x ** 2 + 1])
    X = QQ.old_poly_ring(x, order='ilex').convert(x)
    assert e == [X, X ** 2 + 1]
    assert e == [x, x ** 2 + 1]
    assert 2 * e == [2 * x, 2 * x ** 2 + 2]
    assert e * 2 == [2 * x, 2 * x ** 2 + 2]
    assert e / 2 == [x / 2, (x ** 2 + 1) / 2]
    assert x * e == [x ** 2, x ** 3 + x]
    assert e * x == [x ** 2, x ** 3 + x]
    assert e / (1 + x) == [x / (1 + x), (x ** 2 + 1) / (1 + x)]
    assert X * e == [x ** 2, x ** 3 + x]
    assert e * X == [x ** 2, x ** 3 + x]
    M3 = FreeModule(QQ.old_poly_ring(x, y), 2)
    assert M3.convert(e) == M3.convert([x, x ** 2 + 1])
    assert not M3.is_submodule(0)
    assert not M3.is_zero()
    raises(NotImplementedError, lambda: ZZ.old_poly_ring(x).free_module(2))
    raises(NotImplementedError, lambda: FreeModulePolyRing(ZZ, 2))
    raises(CoercionFailed, lambda: M1.convert(QQ.old_poly_ring(x).free_module(3).convert([1, 2, 3])))
    raises(CoercionFailed, lambda: M3.convert(1))