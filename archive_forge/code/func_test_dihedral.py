from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
from sympy.combinatorics.perm_groups import (PermutationGroup,
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
from sympy.testing.pytest import skip, XFAIL, slow
def test_dihedral():
    G = SymmetricGroup(2)
    assert G.is_dihedral
    G = SymmetricGroup(3)
    assert G.is_dihedral
    G = AbelianGroup(2, 2)
    assert G.is_dihedral
    G = CyclicGroup(4)
    assert not G.is_dihedral
    G = AbelianGroup(3, 5)
    assert not G.is_dihedral
    G = AbelianGroup(2)
    assert G.is_dihedral
    G = AbelianGroup(6)
    assert not G.is_dihedral
    G = PermutationGroup(Permutation(1, 5)(2, 4), Permutation(0, 1)(3, 4)(2, 5))
    assert G.is_dihedral
    G = PermutationGroup(Permutation(1, 6)(2, 5)(3, 4), Permutation(0, 1, 2, 3, 4, 5, 6))
    assert G.is_dihedral
    G = PermutationGroup(Permutation(0, 1), Permutation(0, 2), Permutation(0, 3))
    assert not G.is_dihedral
    G = PermutationGroup(Permutation(1, 6)(2, 5)(3, 4), Permutation(2, 0)(3, 6)(4, 5), Permutation(0, 1, 2, 3, 4, 5, 6))
    assert G.is_dihedral