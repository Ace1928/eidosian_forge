from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
from sympy.combinatorics.perm_groups import (PermutationGroup,
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
from sympy.testing.pytest import skip, XFAIL, slow
def test_coset_factor():
    a = Permutation([0, 2, 1])
    G = PermutationGroup([a])
    c = Permutation([2, 1, 0])
    assert not G.coset_factor(c)
    assert G.coset_rank(c) is None
    a = Permutation([2, 0, 1, 3, 4, 5])
    b = Permutation([2, 1, 3, 4, 5, 0])
    g = PermutationGroup([a, b])
    assert g.order() == 360
    d = Permutation([1, 0, 2, 3, 4, 5])
    assert not g.coset_factor(d.array_form)
    assert not g.contains(d)
    assert Permutation(2) in G
    c = Permutation([1, 0, 2, 3, 5, 4])
    v = g.coset_factor(c, True)
    tr = g.basic_transversals
    p = Permutation.rmul(*[tr[i][v[i]] for i in range(len(g.base))])
    assert p == c
    v = g.coset_factor(c)
    p = Permutation.rmul(*v)
    assert p == c
    assert g.contains(c)
    G = PermutationGroup([Permutation([2, 1, 0])])
    p = Permutation([1, 0, 2])
    assert G.coset_factor(p) == []