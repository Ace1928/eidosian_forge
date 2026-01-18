from sympy.polys.agca.modules import FreeModule, ModuleOrder, FreeModulePolyRing
from sympy.polys import CoercionFailed, QQ, lex, grlex, ilex, ZZ
from sympy.abc import x, y, z
from sympy.testing.pytest import raises
from sympy.core.numbers import Rational
def test_ModulesQuotientRing():
    R = QQ.old_poly_ring(x, y, order=(('lex', x), ('ilex', y))) / [x ** 2 + 1]
    M1 = R.free_module(2)
    assert M1 == R.free_module(2)
    assert M1 != QQ.old_poly_ring(x).free_module(2)
    assert M1 != R.free_module(3)
    assert [x, 1] in M1
    assert [x] not in M1
    assert [1 / (R.convert(x) + 1), 2] in M1
    assert [1, 2 / (1 + y)] in M1
    assert [1, 2 / y] not in M1
    assert M1.convert([x ** 2, y]) == [-1, y]
    F = R.free_module(3)
    Fd = F.submodule([x ** 2, 0, 0], [1, 2, 0], [1, 2, 3])
    M = F.submodule([x ** 2 + y ** 2, 1, 0], [x, y, 1])
    assert F == Fd
    assert Fd == F
    assert F != M
    assert M != F
    assert Fd != M
    assert M != Fd
    assert Fd == F.submodule(*F.basis())
    assert Fd.is_full_module()
    assert not M.is_full_module()
    assert not Fd.is_zero()
    assert not M.is_zero()
    assert Fd.submodule().is_zero()
    assert M.contains([x ** 2 + y ** 2 + x, -x ** 2 + y, 1])
    assert not M.contains([x ** 2 + y ** 2 + x, 1 + y, 2])
    assert M.contains([y ** 2, 1 - x * y, -x])
    assert F.submodule([x, 0, 0]) == F.submodule([1, 0, 0])
    assert not F.submodule([y, 0, 0]) == F.submodule([1, 0, 0])
    assert F.submodule([1, 0, 0], [0, 1, 0]).union(F.submodule([0, 0, 1])) == F
    assert not M.is_submodule(0)