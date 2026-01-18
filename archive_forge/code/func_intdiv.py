from functools import reduce
from operator import mul, add
def intdiv(p, q):
    """Integer division which rounds toward zero

    Examples
    --------
    >>> intdiv(3, 2)
    1
    >>> intdiv(-3, 2)
    -1
    >>> -3 // 2
    -2

    """
    r = p // q
    if r < 0 and q * r != p:
        r += 1
    return r