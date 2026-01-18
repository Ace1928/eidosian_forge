from ..sage_helper import _within_sage
from . import exhaust
from .links_base import Link
def num_Pn(n):
    """
    An element of Jake's P_{0, n} of planar tangles from 0 points to n
    points will be a PerfectMatching of [0,..,n-1] where
    is_noncrossing is True.
    """
    return len([m for m in PerfectMatchings(n) if m.is_noncrossing()])