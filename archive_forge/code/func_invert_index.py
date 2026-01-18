from collections import defaultdict
import numpy as np
from .. import util
from ..dimension import dimension_name
from ..element import Element
from ..ndmapping import NdMapping, item_check, sorted_context
from .dictionary import DictInterface
from .interface import DataError, Interface
from .util import dask_array_module, finite_range, get_array_types, is_dask
@classmethod
def invert_index(cls, index, length):
    if np.isscalar(index):
        return length - index
    elif isinstance(index, slice):
        start, stop = (index.start, index.stop)
        new_start, new_stop = (None, None)
        if start is not None:
            new_stop = length - start
        if stop is not None:
            new_start = length - stop
        return slice(new_start - 1, new_stop - 1)
    elif isinstance(index, util.Iterable):
        new_index = []
        for ind in index:
            new_index.append(length - ind)
    return new_index