from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
@pytest.mark.parametrize('test_geom', ((-1, -1, -0.5), -0.5, None, Point(0, 0)))
def test_intersection_invalid_bounds_tuple(self, test_geom):
    """Tests the `intersection` method with invalid inputs."""
    if compat.USE_PYGEOS:
        with pytest.raises(TypeError):
            self.df.sindex.intersection(test_geom)
    else:
        with pytest.raises((TypeError, Exception)):
            self.df.sindex.intersection(test_geom)