import operator
import itertools
from pyomo.common.dependencies import numpy, numpy_available, scipy, scipy_available
def is_positive_power_of_two(x):
    """Checks if a number is a nonzero and positive power of 2"""
    if x <= 0:
        return False
    else:
        return x & x - 1 == 0