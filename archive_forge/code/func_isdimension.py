import numpy as np
from .util import collect, dshape
from .internal_utils import remove
from .coretypes import (DataShape, Fixed, Var, Ellipsis, Record, Unit,
from .typesets import floating, boolean
def isdimension(ds):
    """ Is a component a dimension?

    >>> from datashader.datashape import int32
    >>> isdimension(Fixed(10))
    True
    >>> isdimension(Var())
    True
    >>> isdimension(int32)
    False
    """
    return isinstance(ds, dimension_types)