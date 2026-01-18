import numpy as np
from shapely.geometry import Point
from shapely.wkt import loads
import geopandas
import pytest
from pandas.testing import assert_series_equal
@pytest.fixture
def geoseries_points():
    p1 = Point(1, 2)
    p2 = Point(2, 3)
    p3 = Point(3, 4)
    p4 = Point(4, 1)
    return geopandas.GeoSeries([p1, p2, p3, p4])