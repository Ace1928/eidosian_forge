import os
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, GeometryCollection, box
import geopandas
from geopandas import GeoDataFrame, GeoSeries, overlay, read_file
from geopandas._compat import PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
def test_a_identity_b(self):
    df_result = overlay(self.layer_a, self.layer_b, how='identity')
    assert_geodataframe_equal(df_result, self.a_identity_b)