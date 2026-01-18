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
@pytest.mark.parametrize('return_distance', [True, False])
@pytest.mark.parametrize('return_all,max_distance,expected', [(True, None, ([[0, 0, 1], [0, 1, 5]], [sqrt(0.5), sqrt(0.5), sqrt(50)])), (False, None, ([[0, 1], [0, 5]], [sqrt(0.5), sqrt(50)])), (True, 1, ([[0, 0], [0, 1]], [sqrt(0.5), sqrt(0.5)])), (False, 1, ([[0], [0]], [sqrt(0.5)]))])
def test_nearest_max_distance(self, expected, max_distance, return_all, return_distance):
    geoms = mod.points(np.arange(10), np.arange(10))
    df = geopandas.GeoDataFrame({'geometry': geoms})
    ps = [Point(0.5, 0.5), Point(0, 10)]
    res = df.sindex.nearest(ps, return_all=return_all, max_distance=max_distance, return_distance=return_distance)
    if return_distance:
        assert_array_equal(res[0], expected[0])
        assert_array_equal(res[1], expected[1])
    else:
        assert_array_equal(res, expected[0])