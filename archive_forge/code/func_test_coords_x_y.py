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
def test_coords_x_y():
    na_value = np.nan
    result = P.x
    expected = [p.x if p is not None else na_value for p in points]
    np.testing.assert_allclose(result, expected)
    result = P.y
    expected = [p.y if p is not None else na_value for p in points]
    np.testing.assert_allclose(result, expected)