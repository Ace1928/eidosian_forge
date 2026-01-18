import numpy as np
from numba.core import types, cgutils
from numba.core.datamodel import models
from numba.core.extending import (
from numba.core.imputils import impl_ret_borrowed
@overload_attribute(IndexType, 'is_monotonic_increasing')
def index_is_monotonic_increasing(index):
    """
    Index.is_monotonic_increasing
    """

    def getter(index):
        data = index._data
        if len(data) == 0:
            return True
        u = data[0]
        for v in data:
            if v < u:
                return False
            v = u
        return True
    return getter