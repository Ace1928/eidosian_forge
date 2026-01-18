import numpy as np
from collections import namedtuple
def make_mergesort_impl(wrap, lt=None, is_argsort=False):
    kwargs_lite = dict(no_cpython_wrapper=True, _nrt=False)
    if lt is None:

        @wrap(**kwargs_lite)
        def lt(a, b):
            return a < b
    else:
        lt = wrap(**kwargs_lite)(lt)
    if is_argsort:

        @wrap(**kwargs_lite)
        def lessthan(a, b, vals):
            return lt(vals[a], vals[b])
    else:

        @wrap(**kwargs_lite)
        def lessthan(a, b, vals):
            return lt(a, b)

    @wrap(**kwargs_lite)
    def argmergesort_inner(arr, vals, ws):
        """The actual mergesort function

        Parameters
        ----------
        arr : array [read+write]
            The values being sorted inplace.  For argsort, this is the
            indices.
        vals : array [readonly]
            ``None`` for normal sort.  In argsort, this is the actual array values.
        ws : array [write]
            The workspace.  Must be of size ``arr.size // 2``
        """
        if arr.size > SMALL_MERGESORT:
            mid = arr.size // 2
            argmergesort_inner(arr[:mid], vals, ws)
            argmergesort_inner(arr[mid:], vals, ws)
            for i in range(mid):
                ws[i] = arr[i]
            left = ws[:mid]
            right = arr[mid:]
            out = arr
            i = j = k = 0
            while i < left.size and j < right.size:
                if not lessthan(right[j], left[i], vals):
                    out[k] = left[i]
                    i += 1
                else:
                    out[k] = right[j]
                    j += 1
                k += 1
            while i < left.size:
                out[k] = left[i]
                i += 1
                k += 1
            while j < right.size:
                out[k] = right[j]
                j += 1
                k += 1
        else:
            i = 1
            while i < arr.size:
                j = i
                while j > 0 and lessthan(arr[j], arr[j - 1], vals):
                    arr[j - 1], arr[j] = (arr[j], arr[j - 1])
                    j -= 1
                i += 1

    @wrap(no_cpython_wrapper=True)
    def mergesort(arr):
        """Inplace"""
        ws = np.empty(arr.size // 2, dtype=arr.dtype)
        argmergesort_inner(arr, None, ws)
        return arr

    @wrap(no_cpython_wrapper=True)
    def argmergesort(arr):
        """Out-of-place"""
        idxs = np.arange(arr.size)
        ws = np.empty(arr.size // 2, dtype=idxs.dtype)
        argmergesort_inner(idxs, arr, ws)
        return idxs
    return MergesortImplementation(run_mergesort=argmergesort if is_argsort else mergesort)