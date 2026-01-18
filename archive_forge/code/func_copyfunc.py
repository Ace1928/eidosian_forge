import math
import types
from itertools import permutations
def copyfunc(f, name=None):
    """Returns a deepcopy of a function."""
    return types.FunctionType(f.__code__, f.__globals__, name or f.__name__, f.__defaults__, f.__closure__)