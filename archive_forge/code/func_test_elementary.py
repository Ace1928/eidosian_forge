from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
from sympy.combinatorics.perm_groups import (PermutationGroup,
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
from sympy.testing.pytest import skip, XFAIL, slow
def test_elementary():
    a = Permutation([1, 5, 2, 0, 3, 6, 4])
    G = PermutationGroup([a])
    assert G.is_elementary(7) is False
    a = Permutation(0, 1)(2, 3)
    b = Permutation(0, 2)(3, 1)
    G = PermutationGroup([a, b])
    assert G.is_elementary(2) is True
    c = Permutation(4, 5, 6)
    G = PermutationGroup([a, b, c])
    assert G.is_elementary(2) is False
    G = SymmetricGroup(4).sylow_subgroup(2)
    assert G.is_elementary(2) is False
    H = AlternatingGroup(4).sylow_subgroup(2)
    assert H.is_elementary(2) is True