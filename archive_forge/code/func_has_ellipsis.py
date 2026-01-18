from __future__ import print_function, division, absolute_import
from itertools import chain
import operator
from .. import parser
from .. import type_symbol_table
from ..validation import validate
from .. import coretypes
def has_ellipsis(ds):
    """Returns True if the datashape has an ellipsis

    >>> has_ellipsis(dshape('2 * int'))
    False
    >>> has_ellipsis(dshape('... * int'))
    True
    """
    return has(coretypes.Ellipsis, ds)