from functools import reduce
from operator import mul
import sys
import warnings
from .util.arithmeticdict import ArithmeticDict
from .util.pyutil import NameSpace, deprecated
def is_quantity(arg):
    if arg.__class__.__name__ == 'Quantity':
        return True
    else:
        return False