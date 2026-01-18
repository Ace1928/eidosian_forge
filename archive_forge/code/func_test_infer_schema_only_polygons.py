from collections import OrderedDict
from shapely.geometry import (
import pandas as pd
import pytest
import numpy as np
from geopandas import GeoDataFrame
from geopandas.io.file import infer_schema
def test_infer_schema_only_polygons():
    df = GeoDataFrame(geometry=[city_hall_boundaries, vauquelin_place])
    assert infer_schema(df) == {'geometry': 'Polygon', 'properties': OrderedDict()}