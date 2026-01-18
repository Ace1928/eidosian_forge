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
def test_globe(self):
    rugby_globe = ccrs.Globe(semimajor_axis=9000000, semiminor_axis=9000000, ellipse=None)
    footy_globe = ccrs.Globe(semimajor_axis=1000000, semiminor_axis=1000000, ellipse=None)
    rugby_moll = ccrs.Mollweide(globe=rugby_globe)
    footy_moll = ccrs.Mollweide(globe=footy_globe)
    rugby_pt = rugby_moll.transform_point(10, 10, rugby_moll.as_geodetic())
    footy_pt = footy_moll.transform_point(10, 10, footy_moll.as_geodetic())
    assert_arr_almost_eq(rugby_pt, (1400915, 1741319), decimal=0)
    assert_arr_almost_eq(footy_pt, (155657, 193479), decimal=0)