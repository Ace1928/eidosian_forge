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
def test_align_mixed(self):
    a1 = self.a1
    s2 = pd.Series([1, 2], index=['B', 'C'])
    res1, res2 = a1.align(s2)
    exp2 = pd.Series([np.nan, 1, 2], index=['A', 'B', 'C'])
    assert_series_equal(res2, exp2)