import collections
from numba.core import types
@wrap
def merge_at(ms, keys, values, i):
    """
        Merge the two runs at stack indices i and i+1.

        An updated MergeState is returned.
        """
    n = ms.n
    assert n >= 2
    assert i >= 0
    assert i == n - 2 or i == n - 3
    ssa, na = ms.pending[i]
    ssb, nb = ms.pending[i + 1]
    assert na > 0 and nb > 0
    assert ssa + na == ssb
    ms.pending[i] = MergeRun(ssa, na + nb)
    if i == n - 3:
        ms.pending[i + 1] = ms.pending[i + 2]
    ms = merge_pop(ms)
    k = gallop_right(keys[ssb], keys, ssa, ssa + na, ssa)
    na -= k - ssa
    ssa = k
    if na == 0:
        return ms
    k = gallop_left(keys[ssa + na - 1], keys, ssb, ssb + nb, ssb + nb - 1)
    nb = k - ssb
    if na <= nb:
        return merge_lo(ms, keys, values, ssa, na, ssb, nb)
    else:
        return merge_hi(ms, keys, values, ssa, na, ssb, nb)