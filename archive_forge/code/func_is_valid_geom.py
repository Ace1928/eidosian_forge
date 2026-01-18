import logging
import math
import os
import warnings
import numpy as np
import rasterio
from rasterio import warp
from rasterio._base import DatasetBase
from rasterio._features import _shapes, _sieve, _rasterize, _bounds
from rasterio.dtypes import validate_dtype, can_cast_dtype, get_minimum_dtype, _getnpdtype
from rasterio.enums import MergeAlg
from rasterio.env import ensure_env, GDALVersion
from rasterio.errors import ShapeSkipWarning
from rasterio.rio.helpers import coords
from rasterio.transform import Affine
from rasterio.transform import IDENTITY, guard_transform
from rasterio.windows import Window
def is_valid_geom(geom):
    """
    Checks to see if geometry is a valid GeoJSON geometry type or
    GeometryCollection.  Geometry must be GeoJSON or implement the geo
    interface.

    Geometries must be non-empty, and have at least x, y coordinates.

    Note: only the first coordinate is checked for validity.

    Parameters
    ----------
    geom: an object that implements the geo interface or GeoJSON-like object

    Returns
    -------
    bool: True if object is a valid GeoJSON geometry type
    """
    geom_types = {'Point', 'MultiPoint', 'LineString', 'LinearRing', 'MultiLineString', 'Polygon', 'MultiPolygon'}
    geom = getattr(geom, '__geo_interface__', None) or geom
    try:
        geom_type = geom['type']
        if geom_type not in geom_types.union({'GeometryCollection'}):
            return False
    except (KeyError, TypeError):
        return False
    if geom_type in geom_types:
        if 'coordinates' not in geom:
            return False
        coords = geom['coordinates']
        if geom_type == 'Point':
            return len(coords) >= 2
        if geom_type == 'MultiPoint':
            return len(coords) > 0 and len(coords[0]) >= 2
        if geom_type == 'LineString':
            return len(coords) >= 2 and len(coords[0]) >= 2
        if geom_type == 'LinearRing':
            return len(coords) >= 4 and len(coords[0]) >= 2
        if geom_type == 'MultiLineString':
            return len(coords) > 0 and len(coords[0]) >= 2 and (len(coords[0][0]) >= 2)
        if geom_type == 'Polygon':
            return len(coords) > 0 and len(coords[0]) >= 4 and (len(coords[0][0]) >= 2)
        if geom_type == 'MultiPolygon':
            return len(coords) > 0 and len(coords[0]) > 0 and (len(coords[0][0]) >= 4) and (len(coords[0][0][0]) >= 2)
    if geom_type == 'GeometryCollection':
        if 'geometries' not in geom:
            return False
        if not len(geom['geometries']) > 0:
            return False
        for g in geom['geometries']:
            if not is_valid_geom(g):
                return False
    return True