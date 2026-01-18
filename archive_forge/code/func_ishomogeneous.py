import numpy as np
from .util import collect, dshape
from .internal_utils import remove
from .coretypes import (DataShape, Fixed, Var, Ellipsis, Record, Unit,
from .typesets import floating, boolean
def ishomogeneous(ds):
    """ Does datashape contain only one dtype?

    >>> from datashader.datashape import int32
    >>> ishomogeneous(int32)
    True
    >>> ishomogeneous('var * 3 * string')
    True
    >>> ishomogeneous('var * {name: string, amount: int}')
    False
    """
    ds = dshape(ds)
    return len(set(remove(isdimension, collect(isscalar, ds)))) == 1