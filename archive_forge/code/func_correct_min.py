from .sage_helper import _within_sage
from functools import reduce
import operator
def correct_min(l):
    """
    A version of min that works correctly even when l is a list of
    real intervals.

    This is needed because python's min returns the wrong result
    for intervals, for example:

        sage: from sage.all import RIF
        sage: min(RIF(4,5), RIF(3,6)).endpoints()
        (4.00000000000000, 5.00000000000000)

    when the correct is answer is [3,5].

    The reason is that python's min(x, y) returns either x or y
    depending on whether y < x. The correct implementation returns
    the smallest lower and upper bound across all intervals,
    respectively.
    """
    are_intervals = [is_RealIntervalFieldElement(x) for x in l]
    if any(are_intervals):
        if not all(are_intervals):
            raise TypeError('Trying to compute min of array where some elements are intervals and others are not.')
        for x in l:
            if x.is_NaN():
                raise ValueError('Trying to compute min of array containing NaN interval.')
        return reduce(lambda x, y: x.min(y), l)
    else:
        for x in l:
            if not x == x:
                raise ValueError('Trying to compute min of array containing NaN.')
        return min(l)