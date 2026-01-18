from functools import reduce
from operator import mul
import sys
import warnings
from .util.arithmeticdict import ArithmeticDict
from .util.pyutil import NameSpace, deprecated
def fold_constants(arg):
    if hasattr(arg, 'dimensionality'):
        m = arg.magnitude
        d = 1
        for k, v in arg.dimensionality.items():
            if isinstance(k, pq.UnitConstant):
                m = m * k.simplified ** v
            else:
                d = d * k ** v
        return m * d
    else:
        return arg