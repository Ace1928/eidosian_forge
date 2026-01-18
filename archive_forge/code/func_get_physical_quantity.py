from functools import reduce
from operator import mul
import sys
import warnings
from .util.arithmeticdict import ArithmeticDict
from .util.pyutil import NameSpace, deprecated
@deprecated(use_instead=get_physical_dimensionality, will_be_missing_in='0.8.0')
def get_physical_quantity(value):
    return get_physical_dimensionality(value)