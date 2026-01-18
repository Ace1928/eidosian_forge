from collections import OrderedDict
from shapely.geometry import (
import pandas as pd
import pytest
import numpy as np
from geopandas import GeoDataFrame
from geopandas.io.file import infer_schema
def test_infer_schema_null_geometry_and_2D_point():
    df = GeoDataFrame(geometry=[None, city_hall_entrance])
    assert infer_schema(df) == {'geometry': 'Point', 'properties': OrderedDict()}