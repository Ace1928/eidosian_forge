from collections import OrderedDict
from shapely.geometry import (
import pandas as pd
import pytest
import numpy as np
from geopandas import GeoDataFrame
from geopandas.io.file import infer_schema
def test_infer_schema_only_multipolygons():
    df = GeoDataFrame(geometry=[MultiPolygon((city_hall_boundaries, vauquelin_place))])
    assert infer_schema(df) == {'geometry': 'MultiPolygon', 'properties': OrderedDict()}