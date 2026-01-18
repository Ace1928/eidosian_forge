import collections
from numba.core import types
@wrap
def gallop_right(key, a, start, stop, hint):
    """
        Exactly like gallop_left(), except that if key already exists in a[start:stop],
        finds the position immediately to the right of the rightmost equal value.

        The return value is the int k in start..stop such that

            a[k-1] <= key < a[k]

        The code duplication is massive, but this is enough different given that
        we're sticking to "<" comparisons that it's much harder to follow if
        written as one routine with yet another "left or right?" flag.
        """
    assert stop > start
    assert hint >= start and hint < stop
    n = stop - start
    lastofs = 0
    ofs = 1
    if LT(key, a[hint]):
        maxofs = hint - start + 1
        while ofs < maxofs:
            if LT(key, a[hint - ofs]):
                lastofs = ofs
                ofs = (ofs << 1) + 1
                if ofs <= 0:
                    ofs = maxofs
            else:
                break
        if ofs > maxofs:
            ofs = maxofs
        lastofs, ofs = (hint - ofs, hint - lastofs)
    else:
        maxofs = stop - hint
        while ofs < maxofs:
            if LT(key, a[hint + ofs]):
                break
            else:
                lastofs = ofs
                ofs = (ofs << 1) + 1
                if ofs <= 0:
                    ofs = maxofs
        if ofs > maxofs:
            ofs = maxofs
        lastofs += hint
        ofs += hint
    assert start - 1 <= lastofs and lastofs < ofs and (ofs <= stop)
    lastofs += 1
    while lastofs < ofs:
        m = lastofs + (ofs - lastofs >> 1)
        if LT(key, a[m]):
            ofs = m
        else:
            lastofs = m + 1
    return ofs