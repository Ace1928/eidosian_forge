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
def test_no_warning_if_aligned(self):
    a1, a2 = self.a1.align(self.a2)
    with warnings.catch_warnings(record=True) as record:
        a1.contains(a2)
        self.g1.intersects(self.g2)
        a2.union(a1)
        self.g2.intersection(self.g1)
    user_warnings = [w for w in record if w.category is UserWarning]
    assert not user_warnings, user_warnings[0].message