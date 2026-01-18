import json
import warnings
import numpy as np
import pandas as pd
import shapely.errors
from pandas import DataFrame, Series
from pandas.core.accessor import CachedAccessor
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from pyproj import CRS
from geopandas.array import GeometryArray, GeometryDtype, from_shapely, to_wkb, to_wkt
from geopandas.base import GeoPandasBase, is_geometry_type
from geopandas.geoseries import GeoSeries
import geopandas.io
from geopandas.explore import _explore
from . import _compat as compat
from ._decorator import doc
def set_geometry(self, col, drop=False, inplace=False, crs=None):
    """
        Set the GeoDataFrame geometry using either an existing column or
        the specified input. By default yields a new object.

        The original geometry column is replaced with the input.

        Parameters
        ----------
        col : column label or array
        drop : boolean, default False
            Delete column to be used as the new geometry
        inplace : boolean, default False
            Modify the GeoDataFrame in place (do not create a new object)
        crs : pyproj.CRS, optional
            Coordinate system to use. The value can be anything accepted
            by :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.
            If passed, overrides both DataFrame and col's crs.
            Otherwise, tries to get crs from passed col values or DataFrame.

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> d = {'col1': ['name1', 'name2'], 'geometry': [Point(1, 2), Point(2, 1)]}
        >>> gdf = geopandas.GeoDataFrame(d, crs="EPSG:4326")
        >>> gdf
            col1                 geometry
        0  name1  POINT (1.00000 2.00000)
        1  name2  POINT (2.00000 1.00000)

        Passing an array:

        >>> df1 = gdf.set_geometry([Point(0,0), Point(1,1)])
        >>> df1
            col1                 geometry
        0  name1  POINT (0.00000 0.00000)
        1  name2  POINT (1.00000 1.00000)

        Using existing column:

        >>> gdf["buffered"] = gdf.buffer(2)
        >>> df2 = gdf.set_geometry("buffered")
        >>> df2.geometry
        0    POLYGON ((3.00000 2.00000, 2.99037 1.80397, 2....
        1    POLYGON ((4.00000 1.00000, 3.99037 0.80397, 3....
        Name: buffered, dtype: geometry

        Returns
        -------
        GeoDataFrame

        See also
        --------
        GeoDataFrame.rename_geometry : rename an active geometry column
        """
    if inplace:
        frame = self
    else:
        frame = self.copy()
    to_remove = None
    geo_column_name = self._geometry_column_name
    if geo_column_name is None:
        geo_column_name = 'geometry'
    if isinstance(col, (Series, list, np.ndarray, GeometryArray)):
        level = col
    elif hasattr(col, 'ndim') and col.ndim > 1:
        raise ValueError('Must pass array with one dimension only.')
    else:
        try:
            level = frame[col]
        except KeyError:
            raise ValueError('Unknown column %s' % col)
        except Exception:
            raise
        if isinstance(level, DataFrame):
            raise ValueError('GeoDataFrame does not support setting the geometry column where the column name is shared by multiple columns.')
        if drop:
            to_remove = col
        else:
            geo_column_name = col
    if to_remove:
        del frame[to_remove]
    if not crs:
        crs = getattr(level, 'crs', None)
    if isinstance(level, (GeoSeries, GeometryArray)) and level.crs != crs:
        level = level.copy()
        level.crs = crs
    level = _ensure_geometry(level, crs=crs)
    frame._geometry_column_name = geo_column_name
    frame[geo_column_name] = level
    if not inplace:
        return frame