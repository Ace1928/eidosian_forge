from shapely.geometry import LineString, MultiPoint, Point
from geopandas import GeoSeries
from geopandas.tools import collect
import pytest
def test_collect_mixed_types(self):
    with pytest.raises(ValueError):
        collect([self.p1, self.line1])