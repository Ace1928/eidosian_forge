import collections
from numba.core import types
@wrap
def merge_getmem(ms, need):
    """
        Ensure enough temp memory for 'need' items is available.
        """
    alloced = len(ms.keys)
    if need <= alloced:
        return ms
    while alloced < need:
        alloced = alloced << 1
    temp_keys = make_temp_area(ms.keys, alloced)
    if has_values(ms.keys, ms.values):
        temp_values = make_temp_area(ms.values, alloced)
    else:
        temp_values = temp_keys
    return MergeState(ms.min_gallop, temp_keys, temp_values, ms.pending, ms.n)