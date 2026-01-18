from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
from sympy.combinatorics.perm_groups import (PermutationGroup,
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
from sympy.testing.pytest import skip, XFAIL, slow
def test_is_alt_sym():
    G = DihedralGroup(10)
    assert G.is_alt_sym() is False
    assert G._eval_is_alt_sym_naive() is False
    assert G._eval_is_alt_sym_naive(only_alt=True) is False
    assert G._eval_is_alt_sym_naive(only_sym=True) is False
    S = SymmetricGroup(10)
    assert S._eval_is_alt_sym_naive() is True
    assert S._eval_is_alt_sym_naive(only_alt=True) is False
    assert S._eval_is_alt_sym_naive(only_sym=True) is True
    N_eps = 10
    _random_prec = {'N_eps': N_eps, 0: Permutation([[2], [1, 4], [0, 6, 7, 8, 9, 3, 5]]), 1: Permutation([[1, 8, 7, 6, 3, 5, 2, 9], [0, 4]]), 2: Permutation([[5, 8], [4, 7], [0, 1, 2, 3, 6, 9]]), 3: Permutation([[3], [0, 8, 2, 7, 4, 1, 6, 9, 5]]), 4: Permutation([[8], [4, 7, 9], [3, 6], [0, 5, 1, 2]]), 5: Permutation([[6], [0, 2, 4, 5, 1, 8, 3, 9, 7]]), 6: Permutation([[6, 9, 8], [4, 5], [1, 3, 7], [0, 2]]), 7: Permutation([[4], [0, 2, 9, 1, 3, 8, 6, 5, 7]]), 8: Permutation([[1, 5, 6, 3], [0, 2, 7, 8, 4, 9]]), 9: Permutation([[8], [6, 7], [2, 3, 4, 5], [0, 1, 9]])}
    assert S.is_alt_sym(_random_prec=_random_prec) is True
    A = AlternatingGroup(10)
    assert A._eval_is_alt_sym_naive() is True
    assert A._eval_is_alt_sym_naive(only_alt=True) is True
    assert A._eval_is_alt_sym_naive(only_sym=True) is False
    _random_prec = {'N_eps': N_eps, 0: Permutation([[1, 6, 4, 2, 7, 8, 5, 9, 3], [0]]), 1: Permutation([[1], [0, 5, 8, 4, 9, 2, 3, 6, 7]]), 2: Permutation([[1, 9, 8, 3, 2, 5], [0, 6, 7, 4]]), 3: Permutation([[6, 8, 9], [4, 5], [1, 3, 7, 2], [0]]), 4: Permutation([[8], [5], [4], [2, 6, 9, 3], [1], [0, 7]]), 5: Permutation([[3, 6], [0, 8, 1, 7, 5, 9, 4, 2]]), 6: Permutation([[5], [2, 9], [1, 8, 3], [0, 4, 7, 6]]), 7: Permutation([[1, 8, 4, 7, 2, 3], [0, 6, 9, 5]]), 8: Permutation([[5, 8, 7], [3], [1, 4, 2, 6], [0, 9]]), 9: Permutation([[4, 9, 6], [3, 8], [1, 2], [0, 5, 7]])}
    assert A.is_alt_sym(_random_prec=_random_prec) is False
    G = PermutationGroup(Permutation(1, 3, size=8)(0, 2, 4, 6), Permutation(5, 7, size=8)(0, 2, 4, 6))
    assert G.is_alt_sym() is False
    G = DihedralGroup(10)
    assert G._eval_is_alt_sym_monte_carlo() is False
    G = DihedralGroup(20)
    assert G._eval_is_alt_sym_monte_carlo() is False
    G = DihedralGroup(6)
    G.is_alt_sym()
    assert G.is_alt_sym() is False