import warnings
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import GeometryDtype
from geopandas import _vectorized
def geom_equals(this, that):
    """
    Test for geometric equality. Empty or missing geometries are considered
    equal.

    Parameters
    ----------
    this, that : arrays of Geo objects (or anything that has an `is_empty`
                 attribute)

    Returns
    -------
    bool
        True if all geometries in left equal geometries in right
    """
    return _geom_equals_mask(this, that).all()