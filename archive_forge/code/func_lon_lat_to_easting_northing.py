import operator
import sys
from types import BuiltinFunctionType, BuiltinMethodType, FunctionType, MethodType
import numpy as np
import pandas as pd
import param
from ..core.data import PandasInterface
from ..core.dimension import Dimension
from ..core.util import flatten, resolve_dependent_value, unique_iterator
def lon_lat_to_easting_northing(longitude, latitude):
    """
    Projects the given longitude, latitude values into Web Mercator
    (aka Pseudo-Mercator or EPSG:3857) coordinates.

    Longitude and latitude can be provided as scalars, Pandas columns,
    or Numpy arrays, and will be returned in the same form.  Lists
    or tuples will be converted to Numpy arrays.

    Args:
        longitude
        latitude

    Returns:
        (easting, northing)

    Examples:
       easting, northing = lon_lat_to_easting_northing(-74,40.71)

       easting, northing = lon_lat_to_easting_northing(
           np.array([-74]),np.array([40.71])
       )

       df=pandas.DataFrame(dict(longitude=np.array([-74]),latitude=np.array([40.71])))
       df.loc[:, 'longitude'], df.loc[:, 'latitude'] = lon_lat_to_easting_northing(
           df.longitude,df.latitude
       )
    """
    if isinstance(longitude, (list, tuple)):
        longitude = np.array(longitude)
    if isinstance(latitude, (list, tuple)):
        latitude = np.array(latitude)
    origin_shift = np.pi * 6378137
    easting = longitude * origin_shift / 180.0
    with np.errstate(divide='ignore', invalid='ignore'):
        northing = np.log(np.tan((90 + latitude) * np.pi / 360.0)) * origin_shift / np.pi
    return (easting, northing)