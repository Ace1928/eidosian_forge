from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
def test_empty_geoseries(self):
    """Tests creating a spatial index from an empty GeoSeries."""
    s = GeoSeries(dtype=object)
    assert not s.sindex
    assert len(s.sindex) == 0