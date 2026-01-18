from itertools import zip_longest
from sympy.utilities.enumerative import (
from sympy.utilities.iterables import _set_partitions
def multiset_partitions_baseline(multiplicities, components):
    """Enumerates partitions of a multiset

    Parameters
    ==========

    multiplicities
         list of integer multiplicities of the components of the multiset.

    components
         the components (elements) themselves

    Returns
    =======

    Set of partitions.  Each partition is tuple of parts, and each
    part is a tuple of components (with repeats to indicate
    multiplicity)

    Notes
    =====

    Multiset partitions can be created as equivalence classes of set
    partitions, and this function does just that.  This approach is
    slow and memory intensive compared to the more advanced algorithms
    available, but the code is simple and easy to understand.  Hence
    this routine is strictly for testing -- to provide a
    straightforward baseline against which to regress the production
    versions.  (This code is a simplified version of an earlier
    production implementation.)
    """
    canon = []
    for ct, elem in zip(multiplicities, components):
        canon.extend([elem] * ct)
    cache = set()
    n = len(canon)
    for nc, q in _set_partitions(n):
        rv = [[] for i in range(nc)]
        for i in range(n):
            rv[q[i]].append(canon[i])
        canonical = tuple(sorted([tuple(p) for p in rv]))
        cache.add(canonical)
    return cache