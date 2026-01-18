import collections
from numba.core import types
@wrap
def binarysort(keys, values, lo, hi, start):
    """
        binarysort is the best method for sorting small arrays: it does
        few compares, but can do data movement quadratic in the number of
        elements.
        [lo, hi) is a contiguous slice of a list, and is sorted via
        binary insertion.  This sort is stable.
        On entry, must have lo <= start <= hi, and that [lo, start) is already
        sorted (pass start == lo if you don't know!).
        """
    assert lo <= start and start <= hi
    _has_values = has_values(keys, values)
    if lo == start:
        start += 1
    while start < hi:
        pivot = keys[start]
        l = lo
        r = start
        while l < r:
            p = l + (r - l >> 1)
            if LT(pivot, keys[p]):
                r = p
            else:
                l = p + 1
        for p in range(start, l, -1):
            keys[p] = keys[p - 1]
        keys[l] = pivot
        if _has_values:
            pivot_val = values[start]
            for p in range(start, l, -1):
                values[p] = values[p - 1]
            values[l] = pivot_val
        start += 1