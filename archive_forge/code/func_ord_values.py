from math import isinf, isnan
import itertools
import numpy
def ord_values(_):
    """ Return possible range for ord function. """
    return Interval(0, 255)