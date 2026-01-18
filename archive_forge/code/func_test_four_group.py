from sympy.combinatorics.galois import (
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import (
def test_four_group():
    G = S4TransitiveSubgroups.V.get_perm_group()
    A4 = AlternatingGroup(4)
    assert G.is_subgroup(A4)
    assert G.degree == 4
    assert G.is_transitive()
    assert G.order() == 4
    assert not G.is_cyclic