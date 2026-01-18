import re
from numba.core import types
def mangle_args(argtys):
    """
    Mangle sequence of Numba type objects and arbitrary values.
    """
    return ''.join([mangle_type_or_value(t) for t in argtys])