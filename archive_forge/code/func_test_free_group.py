from sympy.combinatorics.free_groups import free_group, FreeGroup
from sympy.core import Symbol
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_free_group():
    G, a, b, c = free_group('a, b, c')
    assert F.generators == (x, y, z)
    assert x * z ** 2 in F
    assert x in F
    assert y * z ** (-1) in F
    assert (y * z) ** 0 in F
    assert a not in F
    assert a ** 0 not in F
    assert len(F) == 3
    assert str(F) == '<free group on the generators (x, y, z)>'
    assert not F == G
    assert F.order() is oo
    assert F.is_abelian == False
    assert F.center() == {F.identity}
    e, = free_group('')
    assert e.order() == 1
    assert e.generators == ()
    assert e.elements == {e.identity}
    assert e.is_abelian == True