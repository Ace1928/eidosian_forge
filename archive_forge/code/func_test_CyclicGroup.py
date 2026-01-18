from sympy.combinatorics.named_groups import (SymmetricGroup, CyclicGroup,
from sympy.testing.pytest import raises
def test_CyclicGroup():
    G = CyclicGroup(10)
    elements = list(G.generate())
    assert len(elements) == 10
    assert G.derived_subgroup().order() == 1
    assert G.is_abelian is True
    assert G.is_solvable is True
    assert G.is_nilpotent is True
    H = CyclicGroup(1)
    assert H.order() == 1
    L = CyclicGroup(2)
    assert L.order() == 2