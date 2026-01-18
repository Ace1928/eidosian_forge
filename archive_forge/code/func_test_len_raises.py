import numpy as np
import pytest
from shapely import GeometryCollection, LineString, Point, wkt
from shapely.geometry import shape
def test_len_raises(geometrycollection_geojson):
    geom = shape(geometrycollection_geojson)
    with pytest.raises(TypeError):
        len(geom)