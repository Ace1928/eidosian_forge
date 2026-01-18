from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
def test_empty_tree_geometries(self):
    """Tests building sindex with interleaved empty geometries."""
    geoms = [Point(0, 0), None, Point(), Point(1, 1), Point()]
    df = geopandas.GeoDataFrame(geometry=geoms)
    assert df.sindex.query(Point(1, 1))[0] == 3