import collections
from numba.core import types
@wrap
def merge_force_collapse(ms, keys, values):
    """
        Regardless of invariants, merge all runs on the stack until only one
        remains.  This is used at the end of the mergesort.

        An updated MergeState is returned.
        """
    while ms.n > 1:
        pending = ms.pending
        n = ms.n - 2
        if n > 0:
            if pending[n - 1].size < pending[n + 1].size:
                n -= 1
        ms = merge_at(ms, keys, values, n)
    return ms