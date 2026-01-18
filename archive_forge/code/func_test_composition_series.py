from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
from sympy.combinatorics.perm_groups import (PermutationGroup,
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
from sympy.testing.pytest import skip, XFAIL, slow
def test_composition_series():
    a = Permutation(1, 2, 3)
    b = Permutation(1, 2)
    G = PermutationGroup([a, b])
    comp_series = G.composition_series()
    assert comp_series == G.derived_series()
    S = SymmetricGroup(4)
    assert S.composition_series()[0] == S
    assert len(S.composition_series()) == 5
    A = AlternatingGroup(4)
    assert A.composition_series()[0] == A
    assert len(A.composition_series()) == 4
    G = CyclicGroup(8)
    series = G.composition_series()
    assert is_isomorphic(series[1], CyclicGroup(4))
    assert is_isomorphic(series[2], CyclicGroup(2))
    assert series[3].is_trivial