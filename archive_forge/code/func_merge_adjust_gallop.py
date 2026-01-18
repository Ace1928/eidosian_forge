import collections
from numba.core import types
@wrap
def merge_adjust_gallop(ms, new_gallop):
    """
        Modify the MergeState's min_gallop.
        """
    return MergeState(intp(new_gallop), ms.keys, ms.values, ms.pending, ms.n)