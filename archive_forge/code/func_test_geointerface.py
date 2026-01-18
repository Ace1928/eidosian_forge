import numpy as np
import pytest
from shapely import GeometryCollection, LineString, Point, wkt
from shapely.geometry import shape
def test_geointerface(geometrycollection_geojson):
    geom = shape(geometrycollection_geojson)
    assert geom.__geo_interface__ == geometrycollection_geojson