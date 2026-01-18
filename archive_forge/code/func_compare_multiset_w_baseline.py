from itertools import zip_longest
from sympy.utilities.enumerative import (
from sympy.utilities.iterables import _set_partitions
def compare_multiset_w_baseline(multiplicities):
    """
    Enumerates the partitions of multiset with AOCP algorithm and
    baseline implementation, and compare the results.

    """
    letters = 'abcdefghijklmnopqrstuvwxyz'
    bl_partitions = multiset_partitions_baseline(multiplicities, letters)
    aocp_partitions = set()
    for state in multiset_partitions_taocp(multiplicities):
        p1 = tuple(sorted([tuple(p) for p in list_visitor(state, letters)]))
        aocp_partitions.add(p1)
    assert bl_partitions == aocp_partitions