from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
from sympy.combinatorics.perm_groups import (PermutationGroup,
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
from sympy.testing.pytest import skip, XFAIL, slow
def test_conjugacy_class():
    S = SymmetricGroup(4)
    x = Permutation(1, 2, 3)
    C = {Permutation(0, 1, 2, size=4), Permutation(0, 1, 3), Permutation(0, 2, 1, size=4), Permutation(0, 2, 3), Permutation(0, 3, 1), Permutation(0, 3, 2), Permutation(1, 2, 3), Permutation(1, 3, 2)}
    assert S.conjugacy_class(x) == C