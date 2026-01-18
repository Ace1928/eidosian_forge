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
def rename_geometry(self, col, inplace=False):
    """
        Renames the GeoDataFrame geometry column to
        the specified name. By default yields a new object.

        The original geometry column is replaced with the input.

        Parameters
        ----------
        col : new geometry column label
        inplace : boolean, default False
            Modify the GeoDataFrame in place (do not create a new object)

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> d = {'col1': ['name1', 'name2'], 'geometry': [Point(1, 2), Point(2, 1)]}
        >>> df = geopandas.GeoDataFrame(d, crs="EPSG:4326")
        >>> df1 = df.rename_geometry('geom1')
        >>> df1.geometry.name
        'geom1'
        >>> df.rename_geometry('geom1', inplace=True)
        >>> df.geometry.name
        'geom1'

        Returns
        -------
        geodataframe : GeoDataFrame

        See also
        --------
        GeoDataFrame.set_geometry : set the active geometry
        """
    geometry_col = self.geometry.name
    if col in self.columns:
        raise ValueError(f'Column named {col} already exists')
    else:
        if not inplace:
            return self.rename(columns={geometry_col: col}).set_geometry(col, inplace)
        self.rename(columns={geometry_col: col}, inplace=inplace)
        self.set_geometry(col, inplace=inplace)