from math import isinf, isnan
import itertools
import numpy
def max_values(args):
    """ Return possible range for max function. """
    return Interval(max((x.low for x in args)), max((x.high for x in args)))