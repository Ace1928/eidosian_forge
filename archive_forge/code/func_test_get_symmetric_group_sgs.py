from sympy.combinatorics.permutations import Permutation, Perm
from sympy.combinatorics.tensor_can import (perm_af_direct_product, dummy_sgs,
from sympy.combinatorics.testutil import canonicalize_naive, graph_certificate
from sympy.testing.pytest import skip, XFAIL
def test_get_symmetric_group_sgs():
    assert get_symmetric_group_sgs(2) == ([0], [Permutation(3)(0, 1)])
    assert get_symmetric_group_sgs(2, 1) == ([0], [Permutation(0, 1)(2, 3)])
    assert get_symmetric_group_sgs(3) == ([0, 1], [Permutation(4)(0, 1), Permutation(4)(1, 2)])
    assert get_symmetric_group_sgs(3, 1) == ([0, 1], [Permutation(0, 1)(3, 4), Permutation(1, 2)(3, 4)])
    assert get_symmetric_group_sgs(4) == ([0, 1, 2], [Permutation(5)(0, 1), Permutation(5)(1, 2), Permutation(5)(2, 3)])
    assert get_symmetric_group_sgs(4, 1) == ([0, 1, 2], [Permutation(0, 1)(4, 5), Permutation(1, 2)(4, 5), Permutation(2, 3)(4, 5)])