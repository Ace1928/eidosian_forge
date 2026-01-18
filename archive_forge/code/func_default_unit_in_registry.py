from functools import reduce
from operator import mul
import sys
import warnings
from .util.arithmeticdict import ArithmeticDict
from .util.pyutil import NameSpace, deprecated
def default_unit_in_registry(value, registry):
    _dimensionality = get_physical_dimensionality(value)
    if _dimensionality == {}:
        return 1
    return _get_unit_from_registry(_dimensionality, registry)