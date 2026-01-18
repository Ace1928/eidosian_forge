from collections import OrderedDict
from shapely.geometry import (
import pandas as pd
import pytest
import numpy as np
from geopandas import GeoDataFrame
from geopandas.io.file import infer_schema
def test_infer_schema_points_and_multipoints():
    df = GeoDataFrame(geometry=[MultiPoint([city_hall_entrance, city_hall_balcony]), city_hall_balcony])
    assert infer_schema(df) == {'geometry': ['MultiPoint', 'Point'], 'properties': OrderedDict()}