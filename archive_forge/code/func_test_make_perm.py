from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
from sympy.combinatorics.perm_groups import (PermutationGroup,
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
from sympy.testing.pytest import skip, XFAIL, slow
def test_make_perm():
    assert cube.pgroup.make_perm(5, seed=list(range(5))) == Permutation([4, 7, 6, 5, 0, 3, 2, 1])
    assert cube.pgroup.make_perm(7, seed=list(range(7))) == Permutation([6, 7, 3, 2, 5, 4, 0, 1])