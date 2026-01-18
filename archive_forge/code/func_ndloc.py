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
def ndloc(cls, dataset, indices):
    selected = {}
    adjusted_inds = []
    all_scalar = True
    for kd, ind in zip(dataset.kdims[::-1], indices):
        coords = cls.coords(dataset, kd.name, True)
        if np.isscalar(ind):
            ind = [ind]
        else:
            all_scalar = False
        selected[kd.name] = coords[ind]
        adjusted_inds.append(ind)
    for kd in dataset.kdims:
        if kd.name not in selected:
            coords = cls.coords(dataset, kd.name)
            selected[kd.name] = coords
            all_scalar = False
    for d in dataset.dimensions():
        if d in dataset.kdims and (not cls.irregular(dataset, d)):
            continue
        arr = cls.values(dataset, d, flat=False, compute=False)
        if all_scalar and len(dataset.vdims) == 1:
            return arr[tuple((ind[0] for ind in adjusted_inds))]
        selected[d.name] = arr[tuple(adjusted_inds)]
    return tuple((selected[d.name] for d in dataset.dimensions()))