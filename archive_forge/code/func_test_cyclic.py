from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
from sympy.combinatorics.perm_groups import (PermutationGroup,
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
from sympy.testing.pytest import skip, XFAIL, slow
def test_cyclic():
    G = SymmetricGroup(2)
    assert G.is_cyclic
    G = AbelianGroup(3, 7)
    assert G.is_cyclic
    G = AbelianGroup(7, 7)
    assert not G.is_cyclic
    G = AlternatingGroup(3)
    assert G.is_cyclic
    G = AlternatingGroup(4)
    assert not G.is_cyclic
    G = PermutationGroup(Permutation(0, 1, 2), Permutation(0, 2, 1))
    assert G.is_cyclic
    G = PermutationGroup(Permutation(0, 1, 2, 3), Permutation(0, 2)(1, 3))
    assert G.is_cyclic
    G = PermutationGroup(Permutation(3), Permutation(0, 1)(2, 3), Permutation(0, 2)(1, 3), Permutation(0, 3)(1, 2))
    assert G.is_cyclic is False
    G = PermutationGroup(Permutation(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), Permutation(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13))
    assert G.is_cyclic
    assert PermutationGroup._distinct_primes_lemma([3, 5]) is True
    assert PermutationGroup._distinct_primes_lemma([5, 7]) is True
    assert PermutationGroup._distinct_primes_lemma([2, 3]) is None
    assert PermutationGroup._distinct_primes_lemma([3, 5, 7]) is None
    assert PermutationGroup._distinct_primes_lemma([5, 7, 13]) is True
    G = PermutationGroup(Permutation(0, 1, 2, 3), Permutation(0, 2)(1, 3))
    assert G.is_cyclic
    assert G._is_abelian
    G = PermutationGroup(*SymmetricGroup(3).generators)
    assert G.is_cyclic is False
    G = PermutationGroup(Permutation(0, 1, 2, 3), Permutation(4, 5, 6))
    assert G.is_cyclic
    G = PermutationGroup(Permutation(0, 1), Permutation(2, 3), Permutation(4, 5, 6))
    assert G.is_cyclic is False