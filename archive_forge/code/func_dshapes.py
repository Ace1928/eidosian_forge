from __future__ import print_function, division, absolute_import
from itertools import chain
import operator
from .. import parser
from .. import type_symbol_table
from ..validation import validate
from .. import coretypes
def dshapes(*args):
    """
    Parse a bunch of datashapes all at once.

    >>> a, b = dshapes('3 * int32', '2 * var * float64')
    """
    return [dshape(arg) for arg in args]