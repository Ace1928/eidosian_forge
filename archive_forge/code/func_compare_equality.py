from functools import reduce
from operator import mul
import sys
import warnings
from .util.arithmeticdict import ArithmeticDict
from .util.pyutil import NameSpace, deprecated
def compare_equality(a, b):
    """Returns True if two arguments are equal.

    Both arguments need to have the same dimensionality.

    Parameters
    ----------
    a : quantity
    b : quantity

    Examples
    --------
    >>> km, m = default_units.kilometre, default_units.metre
    >>> compare_equality(3*km, 3)
    False
    >>> compare_equality(3*km, 3000*m)
    True

    """
    try:
        a + b
    except TypeError:
        try:
            len(a)
        except TypeError:
            return a == b
        else:
            if len(a) != len(b):
                return False
            return all((compare_equality(_a, _b) for _a, _b in zip(a, b)))
    except ValueError:
        return False
    else:
        return a == b