import numpy as np
from .util import collect, dshape
from .internal_utils import remove
from .coretypes import (DataShape, Fixed, Var, Ellipsis, Record, Unit,
from .typesets import floating, boolean
def launder(ds):
    if isinstance(ds, str):
        ds = dshape(ds)
    if isinstance(ds, DataShape):
        ds = ds.measure
    return getattr(ds, 'ty', ds)