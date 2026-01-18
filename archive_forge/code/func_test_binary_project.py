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
@pytest.mark.parametrize('normalized', [True, False])
def test_binary_project(normalized):
    na_value = np.nan
    lines = [None] + [shapely.geometry.LineString([(random.random(), random.random()) for _ in range(2)]) for _ in range(len(P) - 2)] + [None]
    L = from_shapely(lines)
    result = L.project(P, normalized=normalized)
    expected = [line.project(p, normalized=normalized) if line is not None and p is not None else na_value for p, line in zip(points, lines)]
    np.testing.assert_allclose(result, expected)