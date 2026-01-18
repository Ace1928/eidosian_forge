from itertools import zip_longest
from sympy.utilities.enumerative import (
from sympy.utilities.iterables import _set_partitions
def part_range_filter(partition_iterator, lb, ub):
    """
    Filters (on the number of parts) a multiset partition enumeration

    Arguments
    =========

    lb, and ub are a range (in the Python slice sense) on the lpart
    variable returned from a multiset partition enumeration.  Recall
    that lpart is 0-based (it points to the topmost part on the part
    stack), so if you want to return parts of sizes 2,3,4,5 you would
    use lb=1 and ub=5.
    """
    for state in partition_iterator:
        f, lpart, pstack = state
        if lpart >= lb and lpart < ub:
            yield state