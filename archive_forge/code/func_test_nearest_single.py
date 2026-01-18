from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
@pytest.mark.skipif(not (compat.USE_SHAPELY_20 or (compat.USE_PYGEOS and compat.PYGEOS_GE_010)), reason='PyGEOS >= 0.10 is required to test sindex.nearest')
@pytest.mark.parametrize('return_all', [True, False])
@pytest.mark.parametrize('geometry,expected', [([0.25, 0.25], [[0], [0]]), ([0.75, 0.75], [[0], [1]])])
def test_nearest_single(self, geometry, expected, return_all):
    geoms = mod.points(np.arange(10), np.arange(10))
    df = geopandas.GeoDataFrame({'geometry': geoms})
    p = Point(geometry)
    res = df.sindex.nearest(p, return_all=return_all)
    assert_array_equal(res, expected)
    p = mod.points(geometry)
    res = df.sindex.nearest(p, return_all=return_all)
    assert_array_equal(res, expected)