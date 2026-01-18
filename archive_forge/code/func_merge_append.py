import collections
from numba.core import types
@wrap
def merge_append(ms, run):
    """
        Append a run on the merge stack.
        """
    n = ms.n
    assert n < MAX_MERGE_PENDING
    ms.pending[n] = run
    return MergeState(ms.min_gallop, ms.keys, ms.values, ms.pending, n + 1)