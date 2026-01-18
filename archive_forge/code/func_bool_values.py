from math import isinf, isnan
import itertools
import numpy
def bool_values(_):
    """ Return the range of a boolean value, i.e. [0, 1]. """
    return Interval(0, 1)