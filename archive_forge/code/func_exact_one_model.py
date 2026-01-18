import common_z3 as CM_Z3
import ctypes
from .z3 import *
def exact_one_model(f):
    """
    return True if f has exactly 1 model, False otherwise.

    EXAMPLES:

    >>> x, y = Ints('x y')
    >>> exact_one_model(And(0<=x**y,x <= 0))
    False

    >>> exact_one_model(And(0<=x,x <= 0))
    True

    >>> exact_one_model(And(0<=x,x <= 1))
    False

    >>> exact_one_model(And(0<=x,x <= -1))
    False
    """
    models = get_models(f, k=2)
    if isinstance(models, list):
        return len(models) == 1
    else:
        return False