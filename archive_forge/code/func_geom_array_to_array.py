import sys
from collections import defaultdict
import numpy as np
import pandas as pd
from ..dimension import dimension_name
from ..util import isscalar, unique_array, unique_iterator
from .interface import DataError, Interface
from .multipath import MultiInterface, ensure_ring
from .pandas import PandasInterface
def geom_array_to_array(geom_array, index, expand=False, geom_type=None):
    """Converts spatialpandas extension arrays to a flattened array.

    Args:
        geom: spatialpandas geometry
        index: The column index to return

    Returns:
        Flattened array
    """
    from spatialpandas.geometry import MultiPointArray, PointArray
    if isinstance(geom_array, PointArray):
        return geom_array.y if index else geom_array.x
    arrays = []
    multi_point = isinstance(geom_array, MultiPointArray) or geom_type == 'Point'
    for geom in geom_array:
        array = geom_to_array(geom, index, multi=expand, geom_type=geom_type)
        if expand:
            arrays.extend(array)
            if not multi_point:
                arrays.append([np.nan])
        else:
            arrays.append(array)
    if expand:
        if not multi_point:
            arrays = arrays[:-1]
        return np.concatenate(arrays) if arrays else np.array([])
    else:
        array = np.empty(len(arrays), dtype=object)
        array[:] = arrays
        return array