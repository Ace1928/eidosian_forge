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
def key_select_mask(cls, dataset, values, ind):
    if values.dtype.kind == 'M':
        ind = util.parse_datetime_selection(ind)
    if isinstance(ind, tuple):
        ind = slice(*ind)
    if isinstance(ind, get_array_types()):
        mask = ind
    elif isinstance(ind, slice):
        mask = True
        if ind.start is not None:
            mask &= ind.start <= values
        if ind.stop is not None:
            mask &= values < ind.stop
        if mask is True:
            mask = np.ones(values.shape, dtype=np.bool_)
    elif isinstance(ind, (set, list)):
        iter_slcs = []
        for ik in ind:
            iter_slcs.append(values == ik)
        mask = np.logical_or.reduce(iter_slcs)
    elif callable(ind):
        mask = ind(values)
    elif ind is None:
        mask = None
    else:
        index_mask = values == ind
        if (dataset.ndims == 1 or dataset._binned) and np.sum(index_mask) == 0:
            data_index = np.argmin(np.abs(values - ind))
            mask = np.zeros(len(values), dtype=np.bool_)
            mask[data_index] = True
        else:
            mask = index_mask
    if mask is None:
        mask = np.ones(values.shape, dtype=bool)
    return mask