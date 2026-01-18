import operator
import itertools
from pyomo.common.dependencies import numpy, numpy_available, scipy, scipy_available
def is_nondecreasing(vals):
    """Checks if a list of points is nondecreasing"""
    if len(vals) <= 1:
        return True
    it = iter(vals)
    next(it)
    op = operator.ge
    return all(itertools.starmap(op, zip(it, vals)))