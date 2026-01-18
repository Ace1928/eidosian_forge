from collections import OrderedDict
from shapely.geometry import (
import pandas as pd
import pytest
import numpy as np
from geopandas import GeoDataFrame
from geopandas.io.file import infer_schema
def test_infer_schema_mixed_3D_shape_type():
    df = GeoDataFrame(geometry=[MultiPolygon((city_hall_boundaries, vauquelin_place)), city_hall_boundaries, MultiLineString(city_hall_walls), city_hall_walls[0], MultiPoint([city_hall_entrance, city_hall_balcony]), city_hall_balcony, point_3D])
    assert infer_schema(df) == {'geometry': ['3D Point', 'MultiPolygon', 'Polygon', 'MultiLineString', 'LineString', 'MultiPoint', 'Point'], 'properties': OrderedDict()}