from sympy.combinatorics.named_groups import SymmetricGroup, AlternatingGroup,\
from sympy.combinatorics.testutil import _verify_bsgs, _cmp_perm_lists,\
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.core.random import shuffle
def test_naive_list_centralizer():
    S = SymmetricGroup(3)
    A = AlternatingGroup(3)
    assert _naive_list_centralizer(S, S) == [Permutation([0, 1, 2])]
    assert PermutationGroup(_naive_list_centralizer(S, A)).is_subgroup(A)