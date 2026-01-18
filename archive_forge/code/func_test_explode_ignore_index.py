import json
import os
import random
import re
import shutil
import tempfile
import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from pandas.testing import assert_index_equal
from pyproj import CRS
from shapely.geometry import (
from shapely.geometry.base import BaseGeometry
from geopandas import GeoSeries, GeoDataFrame, read_file, datasets, clip
from geopandas._compat import ignore_shapely2_warnings
from geopandas.array import GeometryArray, GeometryDtype
from geopandas.testing import assert_geoseries_equal, geom_almost_equals
from geopandas.tests.util import geom_equals
from pandas.testing import assert_series_equal
import pytest
def test_explode_ignore_index(self):
    s = GeoSeries([MultiPoint([(0, 0), (1, 1)]), MultiPoint([(2, 2), (3, 3), (4, 4)])])
    s = s.explode(ignore_index=True)
    expected_index = pd.Index(range(len(s)))
    assert_index_equal(s.index, expected_index)
    s = s.explode(index_parts=True, ignore_index=True)
    assert_index_equal(s.index, expected_index)