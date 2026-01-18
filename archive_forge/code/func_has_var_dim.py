from __future__ import print_function, division, absolute_import
from itertools import chain
import operator
from .. import parser
from .. import type_symbol_table
from ..validation import validate
from .. import coretypes
def has_var_dim(ds):
    """Returns True if datashape has a variable dimension

    Note currently treats variable length string as scalars.

    >>> has_var_dim(dshape('2 * int32'))
    False
    >>> has_var_dim(dshape('var * 2 * int32'))
    True
    """
    return has((coretypes.Ellipsis, coretypes.Var), ds)