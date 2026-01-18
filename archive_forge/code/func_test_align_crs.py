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
def test_align_crs(self):
    a1 = self.a1
    a1.crs = 'epsg:4326'
    a2 = self.a2
    a2.crs = 'epsg:31370'
    res1, res2 = a1.align(a2)
    assert res1.crs == 'epsg:4326'
    assert res2.crs == 'epsg:31370'
    a2.crs = None
    res1, res2 = a1.align(a2)
    assert res1.crs == 'epsg:4326'
    assert res2.crs is None