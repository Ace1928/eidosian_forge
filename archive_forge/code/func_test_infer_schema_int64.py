from collections import OrderedDict
from shapely.geometry import (
import pandas as pd
import pytest
import numpy as np
from geopandas import GeoDataFrame
from geopandas.io.file import infer_schema
def test_infer_schema_int64():
    int64col = pd.array([1, np.nan], dtype=pd.Int64Dtype())
    df = GeoDataFrame(geometry=[city_hall_entrance, city_hall_balcony])
    df['int64_column'] = int64col
    assert infer_schema(df) == {'geometry': 'Point', 'properties': OrderedDict([('int64_column', 'int')])}