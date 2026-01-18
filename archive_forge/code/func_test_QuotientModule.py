from sympy.polys.agca.modules import FreeModule, ModuleOrder, FreeModulePolyRing
from sympy.polys import CoercionFailed, QQ, lex, grlex, ilex, ZZ
from sympy.abc import x, y, z
from sympy.testing.pytest import raises
from sympy.core.numbers import Rational
def test_QuotientModule():
    R = QQ.old_poly_ring(x)
    F = R.free_module(3)
    N = F.submodule([1, x, x ** 2])
    M = F / N
    assert M != F
    assert M != N
    assert M == F / [(1, x, x ** 2)]
    assert not M.is_zero()
    assert (F / F.basis()).is_zero()
    SQ = F.submodule([1, x, x ** 2], [2, 0, 0]) / N
    assert SQ == M.submodule([2, x, x ** 2])
    assert SQ != M.submodule([2, 1, 0])
    assert SQ != M
    assert M.is_submodule(SQ)
    assert not SQ.is_full_module()
    raises(ValueError, lambda: N / F)
    raises(ValueError, lambda: F.submodule([2, 0, 0]) / N)
    raises(ValueError, lambda: R.free_module(2) / F)
    raises(CoercionFailed, lambda: F.convert(M.convert([1, x, x ** 2])))
    M1 = F / [[1, 1, 1]]
    M2 = M1.submodule([1, 0, 0], [0, 1, 0])
    assert M1 == M2