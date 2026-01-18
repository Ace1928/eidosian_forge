from sympy.combinatorics.named_groups import SymmetricGroup, AlternatingGroup,\
from sympy.combinatorics.testutil import _verify_bsgs, _cmp_perm_lists,\
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.core.random import shuffle
def test_cmp_perm_lists():
    S = SymmetricGroup(4)
    els = list(S.generate_dimino())
    other = els[:]
    shuffle(other)
    assert _cmp_perm_lists(els, other) is True