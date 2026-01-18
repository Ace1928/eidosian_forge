from collections import OrderedDict
from shapely.geometry import (
import pandas as pd
import pytest
import numpy as np
from geopandas import GeoDataFrame
from geopandas.io.file import infer_schema
def test_infer_schema_mixed_3D_Point():
    df = GeoDataFrame(geometry=[city_hall_balcony, point_3D])
    assert infer_schema(df) == {'geometry': ['3D Point', 'Point'], 'properties': OrderedDict()}