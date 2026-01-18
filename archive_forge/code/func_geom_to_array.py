import sys
from collections import defaultdict
import numpy as np
import pandas as pd
from ..dimension import dimension_name
from ..util import isscalar, unique_array, unique_iterator
from .interface import DataError, Interface
from .multipath import MultiInterface, ensure_ring
from .pandas import PandasInterface
def geom_to_array(geom, index=None, multi=False, geom_type=None):
    """Converts spatialpandas geometry to an array.

    Args:
        geom: spatialpandas geometry
        index: The column index to return
        multi: Whether to concatenate multiple arrays or not

    Returns:
        Array or list of arrays.
    """
    from spatialpandas.geometry import Line, MultiPoint, MultiPolygon, Point, Polygon, Ring
    if isinstance(geom, Point):
        if index is None:
            return np.array([[geom.x, geom.y]])
        arrays = [np.array([geom.y if index else geom.x])]
    elif isinstance(geom, (Polygon, Line, Ring)):
        exterior = geom.data[0] if isinstance(geom, Polygon) else geom.data
        arr = np.array(exterior.as_py()).reshape(-1, 2)
        if isinstance(geom, (Polygon, Ring)):
            arr = ensure_ring(arr)
        arrays = [arr if index is None else arr[:, index]]
    elif isinstance(geom, MultiPoint):
        if index is None:
            arrays = [np.array(geom.buffer_values).reshape(-1, 2)]
        else:
            arrays = [np.array(geom.buffer_values[index::2])]
    else:
        arrays = []
        for g in geom.data:
            exterior = g[0] if isinstance(geom, MultiPolygon) else g
            arr = np.array(exterior.as_py()).reshape(-1, 2)
            if isinstance(geom, MultiPolygon):
                arr = ensure_ring(arr)
            arrays.append(arr if index is None else arr[:, index])
            if geom_type != 'Point':
                arrays.append([[np.nan, np.nan]] if index is None else [np.nan])
        if geom_type != 'Point':
            arrays = arrays[:-1]
    if multi:
        return arrays
    elif len(arrays) == 1:
        return arrays[0]
    else:
        return np.concatenate(arrays)