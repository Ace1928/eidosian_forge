from .sage_helper import _within_sage
from functools import reduce
import operator
def is_RealIntervalFieldElement(x):
    """
        is_RealIntervalFieldElement returns whether x is a real
        interval (constructed with RealIntervalField(precision)(value)).
        """
    return False