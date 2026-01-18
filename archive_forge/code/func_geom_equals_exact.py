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
def geom_equals_exact(self, other, tolerance):
    return self._binary_method('equals_exact', self, other, tolerance=tolerance)