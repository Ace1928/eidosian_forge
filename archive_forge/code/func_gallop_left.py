import collections
from numba.core import types
@wrap
def gallop_left(key, a, start, stop, hint):
    """
        Locate the proper position of key in a sorted vector; if the vector contains
        an element equal to key, return the position immediately to the left of
        the leftmost equal element.  [gallop_right() does the same except returns
        the position to the right of the rightmost equal element (if any).]

        "a" is a sorted vector with stop elements, starting at a[start].
        stop must be > start.

        "hint" is an index at which to begin the search, start <= hint < stop.
        The closer hint is to the final result, the faster this runs.

        The return value is the int k in start..stop such that

            a[k-1] < key <= a[k]

        pretending that a[start-1] is minus infinity and a[stop] is plus infinity.
        IOW, key belongs at index k; or, IOW, the first k elements of a should
        precede key, and the last stop-start-k should follow key.

        See listsort.txt for info on the method.
        """
    assert stop > start
    assert hint >= start and hint < stop
    n = stop - start
    lastofs = 0
    ofs = 1
    if LT(a[hint], key):
        maxofs = stop - hint
        while ofs < maxofs:
            if LT(a[hint + ofs], key):
                lastofs = ofs
                ofs = (ofs << 1) + 1
                if ofs <= 0:
                    ofs = maxofs
            else:
                break
        if ofs > maxofs:
            ofs = maxofs
        lastofs += hint
        ofs += hint
    else:
        maxofs = hint - start + 1
        while ofs < maxofs:
            if LT(a[hint - ofs], key):
                break
            else:
                lastofs = ofs
                ofs = (ofs << 1) + 1
                if ofs <= 0:
                    ofs = maxofs
        if ofs > maxofs:
            ofs = maxofs
        lastofs, ofs = (hint - ofs, hint - lastofs)
    assert start - 1 <= lastofs and lastofs < ofs and (ofs <= stop)
    lastofs += 1
    while lastofs < ofs:
        m = lastofs + (ofs - lastofs >> 1)
        if LT(a[m], key):
            lastofs = m + 1
        else:
            ofs = m
    return ofs