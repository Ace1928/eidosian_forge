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
@pytest.fixture(params=[[ccrs.PlateCarree, {}], [ccrs.PlateCarree, dict(central_longitude=1.23)], [ccrs.NorthPolarStereo, dict(central_longitude=42.5, globe=ccrs.Globe(ellipse='helmert'))], [ccrs.CRS, dict(proj4_params='3088')], [ccrs.epsg, dict(code='3088')]])
def proj_to_copy(request):
    cls, kwargs = request.param
    return cls(**kwargs)