import copy
from io import BytesIO
import os
from pathlib import Path
import pickle
import warnings
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from numpy.testing import assert_array_almost_equal as assert_arr_almost_eq
import pyproj
import pytest
import shapely.geometry as sgeom
import cartopy.crs as ccrs
def test_epsg(self):
    uk = ccrs.epsg(27700)
    assert uk.epsg_code == 27700
    expected_x = (-104009.357, 688806.007)
    expected_y = (-8908.37, 1256558.45)
    expected_threshold = 7928.15
    if pyproj.__proj_version__ >= '9.2.0':
        expected_x = (-104728.764, 688806.007)
        expected_y = (-8908.36, 1256616.32)
        expected_threshold = 7935.34
    assert_almost_equal(uk.x_limits, expected_x, decimal=3)
    assert_almost_equal(uk.y_limits, expected_y, decimal=2)
    assert_almost_equal(uk.threshold, expected_threshold, decimal=2)
    self._check_osgb(uk)