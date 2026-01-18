from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
def test_has_sindex(self):
    """Test the has_sindex method."""
    t1 = Polygon([(0, 0), (1, 0), (1, 1)])
    t2 = Polygon([(0, 0), (1, 1), (0, 1)])
    d = GeoDataFrame({'geom': [t1, t2]}, geometry='geom')
    assert not d.has_sindex
    d.sindex
    assert d.has_sindex
    d.geometry.values._sindex = None
    assert not d.has_sindex
    d.sindex
    assert d.has_sindex
    s = GeoSeries([t1, t2])
    assert not s.has_sindex
    s.sindex
    assert s.has_sindex
    s.values._sindex = None
    assert not s.has_sindex
    s.sindex
    assert s.has_sindex