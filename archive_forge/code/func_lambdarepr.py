from .pycode import (
from .numpy import NumPyPrinter  # NumPyPrinter is imported for backward compatibility
from sympy.core.sorting import default_sort_key
def lambdarepr(expr, **settings):
    """
    Returns a string usable for lambdifying.
    """
    return LambdaPrinter(settings).doprint(expr)