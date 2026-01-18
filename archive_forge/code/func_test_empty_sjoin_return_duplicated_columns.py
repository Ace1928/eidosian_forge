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
def test_empty_sjoin_return_duplicated_columns(self):
    nybb = geopandas.read_file(geopandas.datasets.get_path('nybb'))
    nybb2 = nybb.copy()
    nybb2.geometry = nybb2.translate(200000)
    result = geopandas.sjoin(nybb, nybb2)
    assert 'BoroCode_right' in result.columns
    assert 'BoroCode_left' in result.columns