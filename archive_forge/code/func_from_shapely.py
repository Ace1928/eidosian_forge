import sys
from collections import defaultdict
import numpy as np
import pandas as pd
from ..dimension import dimension_name
from ..util import isscalar, unique_array, unique_iterator
from .interface import DataError, Interface
from .multipath import MultiInterface, ensure_ring
from .pandas import PandasInterface
def from_shapely(data):
    """Converts shapely based data formats to spatialpandas.GeoDataFrame.

    Args:
        data: A list of shapely objects or dictionaries containing
              shapely objects

    Returns:
        A GeoDataFrame containing the shapely geometry data.
    """
    from shapely.geometry.base import BaseGeometry
    from spatialpandas import GeoDataFrame, GeoSeries
    if not data:
        pass
    elif all((isinstance(d, BaseGeometry) for d in data)):
        data = GeoSeries(data).to_frame()
    elif all((isinstance(d, dict) and 'geometry' in d and isinstance(d['geometry'], BaseGeometry) for d in data)):
        new_data = {col: [] for col in data[0]}
        for d in data:
            for col, val in d.items():
                new_data[col].append(val if isscalar(val) or isinstance(val, BaseGeometry) else np.asarray(val))
        new_data['geometry'] = GeoSeries(new_data['geometry'])
        data = GeoDataFrame(new_data)
    return data