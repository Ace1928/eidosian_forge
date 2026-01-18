import random
import numpy as np
import pandas as pd
from pyproj import CRS
import shapely
import shapely.affinity
import shapely.geometry
from shapely.geometry.base import CAP_STYLE, JOIN_STYLE, BaseGeometry
import shapely.wkb
import shapely.wkt
import geopandas
from geopandas.array import (
import geopandas._compat as compat
import pytest
def test_binary_distance():
    attr = 'distance'
    na_value = np.nan
    result = P[:len(T)].distance(T[::-1])
    expected = [getattr(p, attr)(t) if not ((t is None or t.is_empty) or (p is None or p.is_empty)) else na_value for t, p in zip(triangles[::-1], points)]
    np.testing.assert_allclose(result, expected)
    p = points[0]
    result = T.distance(p)
    expected = [getattr(t, attr)(p) if not (t is None or t.is_empty) else na_value for t in triangles]
    np.testing.assert_allclose(result, expected)
    result = T.distance(shapely.geometry.Polygon())
    expected = [na_value] * len(T)
    np.testing.assert_allclose(result, expected)