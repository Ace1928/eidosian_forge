import snappy
import FXrays
def normalize_slope(slope):
    """
    For a tuple (a, b), scale it so that gcd(a,b)=1 and it lies in the
    right half plane.

    >>> normalize_slope( (-10, 5) )
    (2, -1)
    >>> normalize_slope( (-3, 0) )
    (1, 0)

    The corner case of (0, b) is handled like this:

    >>> normalize_slope( (0, -10) )
    (0, 1)
    """
    a, b = weak_normalize_slope(slope)
    if a == b == 0:
        return (0, 0)
    if a < 0:
        a, b = (-a, -b)
    elif a == 0 and b < 0:
        b = -b
    return (a, b)