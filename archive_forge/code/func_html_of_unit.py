from functools import reduce
from operator import mul
import sys
import warnings
from .util.arithmeticdict import ArithmeticDict
from .util.pyutil import NameSpace, deprecated
def html_of_unit(quant):
    """Returns HTML representation of the unit of a quantity

    Examples
    --------
    >>> print(html_of_unit(2*default_units.m**2))
    m<sup>2</sup>

    """
    return quant.dimensionality.html