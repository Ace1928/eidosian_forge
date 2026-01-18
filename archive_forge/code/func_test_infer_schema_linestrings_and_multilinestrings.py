from collections import OrderedDict
from shapely.geometry import (
import pandas as pd
import pytest
import numpy as np
from geopandas import GeoDataFrame
from geopandas.io.file import infer_schema
def test_infer_schema_linestrings_and_multilinestrings():
    df = GeoDataFrame(geometry=[MultiLineString(city_hall_walls), city_hall_walls[0]])
    assert infer_schema(df) == {'geometry': ['MultiLineString', 'LineString'], 'properties': OrderedDict()}