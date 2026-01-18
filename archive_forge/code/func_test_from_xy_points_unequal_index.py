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
def test_from_xy_points_unequal_index(self):
    x = self.landmarks.x
    y = self.landmarks.y
    y.index = -np.arange(len(y))
    crs = self.landmarks.crs
    assert_geoseries_equal(self.landmarks, GeoSeries.from_xy(x, y, index=x.index, crs=crs))
    unindexed_landmarks = self.landmarks.copy()
    unindexed_landmarks.reset_index(inplace=True, drop=True)
    assert_geoseries_equal(unindexed_landmarks, GeoSeries.from_xy(x, y, crs=crs))