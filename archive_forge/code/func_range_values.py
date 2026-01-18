from math import isinf, isnan
import itertools
import numpy
def range_values(args):
    """ Function used to compute returned range value of [x]range function. """
    if len(args) == 1:
        return Interval(0, args[0].high)
    elif len(args) == 2:
        return Interval(args[0].low, args[1].high)
    elif len(args) == 3:
        is_neg = args[2].low < 0
        is_pos = args[2].high > 0
        if is_neg and is_pos:
            return UNKNOWN_RANGE
        elif is_neg:
            return Interval(args[1].low, args[0].high - 1)
        else:
            return Interval(args[0].low, args[1].high - 1)