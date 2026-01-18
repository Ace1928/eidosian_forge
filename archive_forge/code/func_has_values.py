import collections
from numba.core import types
@wrap
def has_values(keys, values):
    return values is not keys