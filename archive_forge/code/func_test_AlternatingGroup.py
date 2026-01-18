from sympy.combinatorics.named_groups import (SymmetricGroup, CyclicGroup,
from sympy.testing.pytest import raises
def test_AlternatingGroup():
    G = AlternatingGroup(5)
    elements = list(G.generate())
    assert len(elements) == 60
    assert [perm.is_even for perm in elements] == [True] * 60
    H = AlternatingGroup(1)
    assert H.order() == 1
    L = AlternatingGroup(2)
    assert L.order() == 1