from sympy.polys.agca.modules import FreeModule, ModuleOrder, FreeModulePolyRing
from sympy.polys import CoercionFailed, QQ, lex, grlex, ilex, ZZ
from sympy.abc import x, y, z
from sympy.testing.pytest import raises
from sympy.core.numbers import Rational
def test_ModuleOrder():
    o1 = ModuleOrder(lex, grlex, False)
    o2 = ModuleOrder(ilex, lex, False)
    assert o1 == ModuleOrder(lex, grlex, False)
    assert (o1 != ModuleOrder(lex, grlex, False)) is False
    assert o1 != o2
    assert o1((1, 2, 3)) == (1, (5, (2, 3)))
    assert o2((1, 2, 3)) == (-1, (2, 3))