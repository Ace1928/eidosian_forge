from collections import OrderedDict
from shapely.geometry import (
import pandas as pd
import pytest
import numpy as np
from geopandas import GeoDataFrame
from geopandas.io.file import infer_schema
def test_infer_schema_null_geometry_all():
    df = GeoDataFrame(geometry=[None, None])
    assert infer_schema(df) == {'geometry': 'Unknown', 'properties': OrderedDict()}