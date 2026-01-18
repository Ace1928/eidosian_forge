from sympy.combinatorics.permutations import Permutation, Perm
from sympy.combinatorics.tensor_can import (perm_af_direct_product, dummy_sgs,
from sympy.combinatorics.testutil import canonicalize_naive, graph_certificate
from sympy.testing.pytest import skip, XFAIL
def test_canonical_free():
    g = Permutation([2, 1, 3, 0, 4, 5])
    dummies = [[2, 3]]
    can = canonicalize(g, dummies, [None], ([], [Permutation(3)], 2, 0))
    assert can == [3, 0, 2, 1, 4, 5]