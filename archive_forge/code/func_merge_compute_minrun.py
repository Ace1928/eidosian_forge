import collections
from numba.core import types
@wrap
def merge_compute_minrun(n):
    """
        Compute a good value for the minimum run length; natural runs shorter
        than this are boosted artificially via binary insertion.

        If n < 64, return n (it's too small to bother with fancy stuff).
        Else if n is an exact power of 2, return 32.
        Else return an int k, 32 <= k <= 64, such that n/k is close to, but
        strictly less than, an exact power of 2.

        See listsort.txt for more info.
        """
    r = 0
    assert n >= 0
    while n >= 64:
        r |= n & 1
        n >>= 1
    return n + r