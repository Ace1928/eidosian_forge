import operator
import itertools
from pyomo.common.dependencies import numpy, numpy_available, scipy, scipy_available
def log2floor(n):
    """Computes the exact value of floor(log2(n)) without
    using floating point calculations. Input argument must
    be a positive integer."""
    assert n > 0
    return n.bit_length() - 1