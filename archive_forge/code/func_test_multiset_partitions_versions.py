from itertools import zip_longest
from sympy.utilities.enumerative import (
from sympy.utilities.iterables import _set_partitions
def test_multiset_partitions_versions():
    """Compares Knuth-based versions of multiset_partitions"""
    multiplicities = [5, 2, 2, 1]
    m = MultisetPartitionTraverser()
    for s1, s2 in zip_longest(m.enum_all(multiplicities), multiset_partitions_taocp(multiplicities)):
        assert compare_multiset_states(s1, s2)