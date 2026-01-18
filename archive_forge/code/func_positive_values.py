from math import isinf, isnan
import itertools
import numpy
def positive_values(_):
    """ Return a positive range without upper bound. """
    return Interval(0, float('inf'))