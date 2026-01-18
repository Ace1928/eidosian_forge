import warnings
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import GeometryDtype
from geopandas import _vectorized
def geom_almost_equals(this, that):
    """
    Test for 'almost' geometric equality. Empty or missing geometries
    considered equal.

    This method allows small difference in the coordinates, but this
    requires coordinates be in the same order for all components of a geometry.

    Parameters
    ----------
    this, that : arrays of Geo objects (or anything that has an `is_empty`
                 property)

    Returns
    -------
    bool
        True if all geometries in left almost equal geometries in right
    """
    if isinstance(this, GeoDataFrame) and isinstance(that, GeoDataFrame):
        this = this.geometry
        that = that.geometry
    return _geom_almost_equals_mask(this, that).all()