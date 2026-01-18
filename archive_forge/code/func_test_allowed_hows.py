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
@pytest.mark.parametrize('how_kwargs', ({}, {'how': 'inner'}, {'how': 'left'}, {'how': 'right'}))
def test_allowed_hows(self, how_kwargs):
    left = geopandas.GeoDataFrame({'geometry': []})
    right = geopandas.GeoDataFrame({'geometry': []})
    sjoin_nearest(left, right, **how_kwargs)