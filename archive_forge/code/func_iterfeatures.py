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
def iterfeatures(self, na='null', show_bbox=False, drop_id=False):
    """
        Returns an iterator that yields feature dictionaries that comply with
        __geo_interface__

        Parameters
        ----------
        na : str, optional
            Options are {'null', 'drop', 'keep'}, default 'null'.
            Indicates how to output missing (NaN) values in the GeoDataFrame

            - null: output the missing entries as JSON null
            - drop: remove the property from the feature. This applies to each feature individually so that features may have different properties
            - keep: output the missing entries as NaN

        show_bbox : bool, optional
            Include bbox (bounds) in the geojson. Default False.
        drop_id : bool, default: False
            Whether to retain the index of the GeoDataFrame as the id property
            in the generated GeoJSON. Default is False, but may want True
            if the index is just arbitrary row numbers.

        Examples
        --------

        >>> from shapely.geometry import Point
        >>> d = {'col1': ['name1', 'name2'], 'geometry': [Point(1, 2), Point(2, 1)]}
        >>> gdf = geopandas.GeoDataFrame(d, crs="EPSG:4326")
        >>> gdf
            col1                 geometry
        0  name1  POINT (1.00000 2.00000)
        1  name2  POINT (2.00000 1.00000)

        >>> feature = next(gdf.iterfeatures())
        >>> feature
        {'id': '0', 'type': 'Feature', 'properties': {'col1': 'name1'}, 'geometry': {'type': 'Point', 'coordinates': (1.0, 2.0)}}
        """
    if na not in ['null', 'drop', 'keep']:
        raise ValueError('Unknown na method {0}'.format(na))
    if self._geometry_column_name not in self:
        raise AttributeError("No geometry data set (expected in column '%s')." % self._geometry_column_name)
    ids = np.array(self.index, copy=False)
    geometries = np.array(self[self._geometry_column_name], copy=False)
    if not self.columns.is_unique:
        raise ValueError('GeoDataFrame cannot contain duplicated column names.')
    properties_cols = self.columns.drop(self._geometry_column_name)
    if len(properties_cols) > 0:
        properties_cols = self[properties_cols]
        properties = properties_cols.astype(object)
        na_mask = pd.isna(properties_cols).values
        if na == 'null':
            properties[na_mask] = None
        for i, row in enumerate(properties.values):
            geom = geometries[i]
            if na == 'drop':
                na_mask_row = na_mask[i]
                properties_items = {k: v for k, v, na in zip(properties_cols, row, na_mask_row) if not na}
            else:
                properties_items = dict(zip(properties_cols, row))
            if drop_id:
                feature = {}
            else:
                feature = {'id': str(ids[i])}
            feature['type'] = 'Feature'
            feature['properties'] = properties_items
            feature['geometry'] = mapping(geom) if geom else None
            if show_bbox:
                feature['bbox'] = geom.bounds if geom else None
            yield feature
    else:
        for fid, geom in zip(ids, geometries):
            if drop_id:
                feature = {}
            else:
                feature = {'id': str(fid)}
            feature['type'] = 'Feature'
            feature['properties'] = {}
            feature['geometry'] = mapping(geom) if geom else None
            if show_bbox:
                feature['bbox'] = geom.bounds if geom else None
            yield feature