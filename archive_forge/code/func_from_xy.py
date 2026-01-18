from __future__ import annotations
import json
import typing
from typing import Optional, Any, Callable, Dict
import warnings
import numpy as np
import pandas as pd
from pandas import Series, MultiIndex
from pandas.core.internals import SingleBlockManager
from pyproj import CRS
import shapely
from shapely.geometry.base import BaseGeometry
from shapely.geometry import GeometryCollection
from geopandas.base import GeoPandasBase, _delegate_property
from geopandas.plotting import plot_series
from geopandas.explore import _explore_geoseries
import geopandas
from . import _compat as compat
from ._decorator import doc
from .array import (
from .base import is_geometry_type
@classmethod
def from_xy(cls, x, y, z=None, index=None, crs=None, **kwargs) -> GeoSeries:
    """
        Alternate constructor to create a :class:`~geopandas.GeoSeries` of Point
        geometries from lists or arrays of x, y(, z) coordinates

        In case of geographic coordinates, it is assumed that longitude is captured
        by ``x`` coordinates and latitude by ``y``.

        Parameters
        ----------
        x, y, z : iterable
        index : array-like or Index, optional
            The index for the GeoSeries. If not given and all coordinate inputs
            are Series with an equal index, that index is used.
        crs : value, optional
            Coordinate Reference System of the geometry objects. Can be anything
            accepted by
            :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.
        **kwargs
            Additional arguments passed to the Series constructor,
            e.g. ``name``.

        Returns
        -------
        GeoSeries

        See Also
        --------
        GeoSeries.from_wkt
        points_from_xy

        Examples
        --------

        >>> x = [2.5, 5, -3.0]
        >>> y = [0.5, 1, 1.5]
        >>> s = geopandas.GeoSeries.from_xy(x, y, crs="EPSG:4326")
        >>> s
        0    POINT (2.50000 0.50000)
        1    POINT (5.00000 1.00000)
        2    POINT (-3.00000 1.50000)
        dtype: geometry
        """
    if index is None:
        if isinstance(x, Series) and isinstance(y, Series) and x.index.equals(y.index) and (z is None or (isinstance(z, Series) and x.index.equals(z.index))):
            index = x.index
    return cls(points_from_xy(x, y, z, crs=crs), index=index, crs=crs, **kwargs)