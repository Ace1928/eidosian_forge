import math
from typing import Sequence
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point, Polygon, GeometryCollection
import geopandas
import geopandas._compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, sjoin, sjoin_nearest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
@pytest.mark.parametrize('how', ('outer', 'abcde'))
def test_invalid_hows(self, how: str):
    left = geopandas.GeoDataFrame({'geometry': []})
    right = geopandas.GeoDataFrame({'geometry': []})
    with pytest.raises(ValueError, match='`how` was'):
        sjoin_nearest(left, right, how=how)