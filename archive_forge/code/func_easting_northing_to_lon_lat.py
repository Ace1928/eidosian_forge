import operator
import sys
from types import BuiltinFunctionType, BuiltinMethodType, FunctionType, MethodType
import numpy as np
import pandas as pd
import param
from ..core.data import PandasInterface
from ..core.dimension import Dimension
from ..core.util import flatten, resolve_dependent_value, unique_iterator
def easting_northing_to_lon_lat(easting, northing):
    """
    Projects the given easting, northing values into
    longitude, latitude coordinates.

    easting and northing values are assumed to be in Web Mercator
    (aka Pseudo-Mercator or EPSG:3857) coordinates.

    Args:
        easting
        northing

    Returns:
        (longitude, latitude)
    """
    if isinstance(easting, (list, tuple)):
        easting = np.array(easting)
    if isinstance(northing, (list, tuple)):
        northing = np.array(northing)
    origin_shift = np.pi * 6378137
    longitude = easting * 180.0 / origin_shift
    with np.errstate(divide='ignore'):
        latitude = np.arctan(np.exp(northing * np.pi / origin_shift)) * 360.0 / np.pi - 90
    return (longitude, latitude)