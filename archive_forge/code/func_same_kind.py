import numpy as np
def same_kind(src, dest):
    """
    Whether the *src* and *dest* units are of the same kind.
    """
    return (DATETIME_UNITS[src] < 5) == (DATETIME_UNITS[dest] < 5)