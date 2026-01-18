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
def test_geom_equals_align(self):
    with pytest.warns(UserWarning, match='The indices .+ different'):
        a = self.a1.geom_equals(self.a2, align=True)
    exp = pd.Series([False, True, False], index=['A', 'B', 'C'])
    assert_series_equal(a, exp)
    a = self.a1.geom_equals(self.a2, align=False)
    exp = pd.Series([False, False], index=['A', 'B'])
    assert_series_equal(a, exp)