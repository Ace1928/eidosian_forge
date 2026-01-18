from math import isinf, isnan
import itertools
import numpy
def min_values(args):
    """ Return possible range for min function. """
    return Interval(min((x.low for x in args)), min((x.high for x in args)))