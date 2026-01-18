import numpy as np
from numba.core import types, cgutils
from numba.core.datamodel import models
from numba.core.extending import (
from numba.core.imputils import impl_ret_borrowed
@lower_builtin('__array_wrap__', SeriesType, types.Array)
def series_wrap_array(context, builder, sig, args):
    src = make_series(context, builder, sig.args[0], value=args[0])
    dest = make_series(context, builder, sig.return_type)
    dest.values = args[1]
    dest.index = src.index
    return impl_ret_borrowed(context, builder, sig.return_type, dest._getvalue())