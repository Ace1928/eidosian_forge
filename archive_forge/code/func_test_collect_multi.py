from shapely.geometry import LineString, MultiPoint, Point
from geopandas import GeoSeries
from geopandas.tools import collect
import pytest
def test_collect_multi(self):
    result = collect(self.mp1)
    assert self.mp1.equals(result)