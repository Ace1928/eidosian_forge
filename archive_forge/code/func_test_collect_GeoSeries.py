from shapely.geometry import LineString, MultiPoint, Point
from geopandas import GeoSeries
from geopandas.tools import collect
import pytest
def test_collect_GeoSeries(self):
    s = GeoSeries([self.p1, self.p2, self.p3])
    result = collect(s)
    assert self.mpc.equals(result)