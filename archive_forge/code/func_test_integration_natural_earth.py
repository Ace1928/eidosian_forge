from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
@pytest.mark.parametrize('predicate, expected_shape', [(None, (2, 471)), ('intersects', (2, 213)), ('within', (2, 213)), ('contains', (2, 0)), ('overlaps', (2, 0)), ('crosses', (2, 0)), ('touches', (2, 0))])
def test_integration_natural_earth(self, predicate, expected_shape):
    """Tests output sizes for the naturalearth datasets."""
    world = read_file(datasets.get_path('naturalearth_lowres'))
    capitals = read_file(datasets.get_path('naturalearth_cities'))
    res = world.sindex.query(capitals.geometry, predicate)
    assert res.shape == expected_shape