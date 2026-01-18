from collections import OrderedDict
from shapely.geometry import (
import pandas as pd
import pytest
import numpy as np
from geopandas import GeoDataFrame
from geopandas.io.file import infer_schema
def test_infer_schema_mixed_3D_linestring():
    df = GeoDataFrame(geometry=[city_hall_walls[0], linestring_3D])
    assert infer_schema(df) == {'geometry': ['3D LineString', 'LineString'], 'properties': OrderedDict()}