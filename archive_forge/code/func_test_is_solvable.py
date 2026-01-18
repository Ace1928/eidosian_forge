from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
from sympy.combinatorics.perm_groups import (PermutationGroup,
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
from sympy.testing.pytest import skip, XFAIL, slow
def test_is_solvable():
    a = Permutation([1, 2, 0])
    b = Permutation([1, 0, 2])
    G = PermutationGroup([a, b])
    assert G.is_solvable
    G = PermutationGroup([a])
    assert G.is_solvable
    a = Permutation([1, 2, 3, 4, 0])
    b = Permutation([1, 0, 2, 3, 4])
    G = PermutationGroup([a, b])
    assert not G.is_solvable
    P = SymmetricGroup(10)
    S = P.sylow_subgroup(3)
    assert S.is_solvable