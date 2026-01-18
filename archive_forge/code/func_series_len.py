import numpy as np
from numba.core import types, cgutils
from numba.core.datamodel import models
from numba.core.extending import (
from numba.core.imputils import impl_ret_borrowed
@overload(len)
def series_len(series):
    """
    len(Series)
    """
    if isinstance(series, SeriesType):

        def len_impl(series):
            return len(series._values)
        return len_impl