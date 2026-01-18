from sympy.combinatorics.named_groups import (SymmetricGroup, CyclicGroup,
from sympy.testing.pytest import raises
def test_DihedralGroup():
    G = DihedralGroup(6)
    elements = list(G.generate())
    assert len(elements) == 12
    assert G.is_transitive() is True
    assert G.is_abelian is False
    assert G.is_solvable is True
    assert G.is_nilpotent is False
    H = DihedralGroup(1)
    assert H.order() == 2
    L = DihedralGroup(2)
    assert L.order() == 4
    assert L.is_abelian is True
    assert L.is_nilpotent is True