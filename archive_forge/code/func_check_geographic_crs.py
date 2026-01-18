import numbers
import operator
import warnings
import inspect
from functools import lru_cache
import numpy as np
import pandas as pd
from pandas.api.extensions import (
import shapely
import shapely.affinity
import shapely.geometry
from shapely.geometry.base import BaseGeometry
import shapely.ops
import shapely.wkt
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from . import _compat as compat
from . import _vectorized as vectorized
from .sindex import _get_sindex_class
def check_geographic_crs(self, stacklevel):
    """Check CRS and warn if the planar operation is done in a geographic CRS"""
    if self.crs and self.crs.is_geographic:
        warnings.warn("Geometry is in a geographic CRS. Results from '{}' are likely incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before this operation.\n".format(inspect.stack()[1].function), UserWarning, stacklevel=stacklevel)