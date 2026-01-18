from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
from sympy.combinatorics.perm_groups import (PermutationGroup,
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
from sympy.testing.pytest import skip, XFAIL, slow
def test_minimal_blocks():
    P = PermutationGroup(Permutation(1, 5)(2, 4), Permutation(0, 1, 2, 3, 4, 5))
    assert P.minimal_blocks() == [[0, 1, 0, 1, 0, 1], [0, 1, 2, 0, 1, 2]]
    P = SymmetricGroup(5)
    assert P.minimal_blocks() == [[0] * 5]
    P = PermutationGroup(Permutation(0, 3))
    assert P.minimal_blocks() is False