from sympy.combinatorics.permutations import Permutation, Perm
from sympy.combinatorics.tensor_can import (perm_af_direct_product, dummy_sgs,
from sympy.combinatorics.testutil import canonicalize_naive, graph_certificate
from sympy.testing.pytest import skip, XFAIL
def test_dummy_sgs():
    a = dummy_sgs([1, 2], 0, 4)
    assert a == [[0, 2, 1, 3, 4, 5]]
    a = dummy_sgs([2, 3, 4, 5], 0, 8)
    assert a == [x._array_form for x in [Perm(9)(2, 3), Perm(9)(4, 5), Perm(9)(2, 4)(3, 5)]]
    a = dummy_sgs([2, 3, 4, 5], 1, 8)
    assert a == [x._array_form for x in [Perm(2, 3)(8, 9), Perm(4, 5)(8, 9), Perm(9)(2, 4)(3, 5)]]