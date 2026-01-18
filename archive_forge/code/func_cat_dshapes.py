from __future__ import print_function, division, absolute_import
from itertools import chain
import operator
from .. import parser
from .. import type_symbol_table
from ..validation import validate
from .. import coretypes
def cat_dshapes(dslist):
    """
    Concatenates a list of dshapes together along
    the first axis. Raises an error if there is
    a mismatch along another axis or the measures
    are different.

    Requires that the leading dimension be a known
    size for all data shapes.
    TODO: Relax this restriction to support
          streaming dimensions.

    >>> cat_dshapes(dshapes('10 * int32', '5 * int32'))
    dshape("15 * int32")
    """
    if len(dslist) == 0:
        raise ValueError('Cannot concatenate an empty list of dshapes')
    elif len(dslist) == 1:
        return dslist[0]
    outer_dim_size = operator.index(dslist[0][0])
    inner_ds = dslist[0][1:]
    for ds in dslist[1:]:
        outer_dim_size += operator.index(ds[0])
        if ds[1:] != inner_ds:
            raise ValueError('The datashapes to concatenate much all match after the first dimension (%s vs %s)' % (inner_ds, ds[1:]))
    return coretypes.DataShape(*[coretypes.Fixed(outer_dim_size)] + list(inner_ds))