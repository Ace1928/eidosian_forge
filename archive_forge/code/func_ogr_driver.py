import os
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal
import pytest
from .test_file import FIONA_MARK, PYOGRIO_MARK
@pytest.fixture(params=['GeoJSON', 'ESRI Shapefile', 'GPKG', 'SQLite'])
def ogr_driver(request):
    return request.param