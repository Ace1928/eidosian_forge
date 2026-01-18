from collections import OrderedDict
from shapely.geometry import (
import pandas as pd
import pytest
import numpy as np
from geopandas import GeoDataFrame
from geopandas.io.file import infer_schema
def test_infer_schema_only_3D_Points():
    df = GeoDataFrame(geometry=[point_3D, point_3D])
    assert infer_schema(df) == {'geometry': '3D Point', 'properties': OrderedDict()}