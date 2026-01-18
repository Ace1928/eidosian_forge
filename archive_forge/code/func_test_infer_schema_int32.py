from collections import OrderedDict
from shapely.geometry import (
import pandas as pd
import pytest
import numpy as np
from geopandas import GeoDataFrame
from geopandas.io.file import infer_schema
@pytest.mark.parametrize('array_data,dtype', [([1, 2 ** 31 - 1], np.int32), ([1, np.nan], pd.Int32Dtype())])
def test_infer_schema_int32(array_data, dtype):
    int32col = pd.array(data=array_data, dtype=dtype)
    df = GeoDataFrame(geometry=[city_hall_entrance, city_hall_balcony])
    df['int32_column'] = int32col
    assert infer_schema(df) == {'geometry': 'Point', 'properties': OrderedDict([('int32_column', 'int32')])}