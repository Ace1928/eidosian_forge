import numpy as np
from .util import collect, dshape
from .internal_utils import remove
from .coretypes import (DataShape, Fixed, Var, Ellipsis, Record, Unit,
from .typesets import floating, boolean
def isboolean(ds):
    """ Has a boolean measure

    >>> isboolean('bool')
    True
    >>> isboolean('3 * ?bool')
    True
    >>> isboolean('int')
    False
    """
    return launder(ds) in boolean