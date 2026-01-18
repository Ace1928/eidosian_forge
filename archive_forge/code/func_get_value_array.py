import sys
from collections import defaultdict
import numpy as np
import pandas as pd
from ..dimension import dimension_name
from ..util import isscalar, unique_array, unique_iterator
from .interface import DataError, Interface
from .multipath import MultiInterface, ensure_ring
from .pandas import PandasInterface
def get_value_array(data, dimension, expanded, keep_index, geom_col, is_points, geom_length=geom_length):
    """Returns an array of values from a GeoDataFrame.

    Args:
        data: GeoDataFrame
        dimension: The dimension to get the values from
        expanded: Whether to expand the value array
        keep_index: Whether to return a Series
        geom_col: The column in the data that contains the geometries
        is_points: Whether the geometries are points
        geom_length: The function used to compute the length of each geometry

    Returns:
        An array containing the values along a dimension
    """
    column = data[dimension.name]
    if keep_index:
        return column
    all_scalar = True
    arrays, scalars = ([], [])
    for i, geom in enumerate(data[geom_col]):
        length = 1 if is_points else geom_length(geom)
        val = column.iloc[i]
        scalar = isscalar(val)
        if scalar:
            val = np.array([val])
        if not scalar and len(unique_array(val)) == 1:
            val = val[:1]
            scalar = True
        all_scalar &= scalar
        scalars.append(scalar)
        if not expanded or not scalar:
            arrays.append(val)
        elif scalar:
            arrays.append(np.full(length, val))
        if expanded and (not is_points) and (not i == len(data[geom_col]) - 1):
            arrays.append(np.array([np.nan]))
    if not len(data):
        return np.array([])
    if expanded:
        return np.concatenate(arrays) if len(arrays) > 1 else arrays[0]
    elif all_scalar and arrays:
        return np.array([a[0] for a in arrays])
    else:
        array = np.empty(len(arrays), dtype=object)
        array[:] = [a[0] if s else a for s, a in zip(scalars, arrays)]
        return array