from .links_base import Strand, Crossing, Link
import random
import collections
def is_range(L):
    """
    >>> is_range([2, 3, 4]), is_range([2, 3, 5])
    (True, False)
    """
    return L == list(range(min(L), max(L) + 1))