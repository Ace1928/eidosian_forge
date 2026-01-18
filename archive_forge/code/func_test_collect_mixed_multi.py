from shapely.geometry import LineString, MultiPoint, Point
from geopandas import GeoSeries
from geopandas.tools import collect
import pytest
def test_collect_mixed_multi(self):
    with pytest.raises(ValueError):
        collect([self.mpc, self.mp1])