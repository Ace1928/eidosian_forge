from shapely.geometry import LineString, MultiPoint, Point
from geopandas import GeoSeries
from geopandas.tools import collect
import pytest
def test_collect_single_force_multi(self):
    result = collect(self.p1, multi=True)
    expected = MultiPoint([self.p1])
    assert expected.equals(result)