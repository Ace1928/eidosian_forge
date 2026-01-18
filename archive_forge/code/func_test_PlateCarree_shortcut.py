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
def test_PlateCarree_shortcut():
    central_lons = [[0, 0], [0, 180], [0, 10], [10, 0], [-180, 180], [180, -180]]
    target = [([[-180, -180], [-180, 180]], 0), ([[-180, 0], [0, 180]], 180), ([[-180, -170], [-170, 180]], 10), ([[-180, 170], [170, 180]], -10), ([[-180, 180], [180, 180]], 360), ([[-180, -180], [-180, 180]], -360)]
    assert len(target) == len(central_lons)
    for expected, (s_lon0, t_lon0) in zip(target, central_lons):
        expected_bboxes, expected_offset = expected
        src = ccrs.PlateCarree(central_longitude=s_lon0)
        target = ccrs.PlateCarree(central_longitude=t_lon0)
        bbox, offset = src._bbox_and_offset(target)
        assert offset == expected_offset
        assert bbox == expected_bboxes