from itertools import zip_longest
from sympy.utilities.enumerative import (
from sympy.utilities.iterables import _set_partitions
def compare_multiset_states(s1, s2):
    """compare for equality two instances of multiset partition states

    This is useful for comparing different versions of the algorithm
    to verify correctness."""
    f1, lpart1, pstack1 = s1
    f2, lpart2, pstack2 = s2
    if lpart1 == lpart2 and f1[0:lpart1 + 1] == f2[0:lpart2 + 1]:
        if pstack1[0:f1[lpart1 + 1]] == pstack2[0:f2[lpart2 + 1]]:
            return True
    return False