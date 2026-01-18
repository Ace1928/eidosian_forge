from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
from sympy.combinatorics.perm_groups import (PermutationGroup,
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
from sympy.testing.pytest import skip, XFAIL, slow
def test_generator_product():
    G = SymmetricGroup(5)
    p = Permutation(0, 2, 3)(1, 4)
    gens = G.generator_product(p)
    assert all((g in G.strong_gens for g in gens))
    w = G.identity
    for g in gens:
        w = g * w
    assert w == p