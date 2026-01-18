from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
from sympy.combinatorics.perm_groups import (PermutationGroup,
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
from sympy.testing.pytest import skip, XFAIL, slow
def test_conjugacy_classes():
    S = SymmetricGroup(3)
    expected = [{Permutation(size=3)}, {Permutation(0, 1, size=3), Permutation(0, 2), Permutation(1, 2)}, {Permutation(0, 1, 2), Permutation(0, 2, 1)}]
    computed = S.conjugacy_classes()
    assert len(expected) == len(computed)
    assert all((e in computed for e in expected))