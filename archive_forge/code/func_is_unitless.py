from functools import reduce
from operator import mul
import sys
import warnings
from .util.arithmeticdict import ArithmeticDict
from .util.pyutil import NameSpace, deprecated
def is_unitless(expr):
    """Returns ``True`` if ``expr`` is unitless, otherwise ``False``

    Examples
    --------
    >>> is_unitless(42)
    True
    >>> is_unitless(42*default_units.kilogram)
    False

    """
    if hasattr(expr, 'dimensionality'):
        if expr.dimensionality == pq.dimensionless:
            return True
        else:
            return expr.simplified.dimensionality == pq.dimensionless.dimensionality
    if isinstance(expr, dict):
        return all((is_unitless(_) for _ in expr.values()))
    elif isinstance(expr, (tuple, list)):
        return all((is_unitless(_) for _ in expr))
    return True