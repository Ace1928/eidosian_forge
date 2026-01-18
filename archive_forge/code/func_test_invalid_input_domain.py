import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
import cartopy.crs as ccrs
def test_invalid_input_domain(self):
    rlon = np.array([270.0])
    rlat = np.array([0.0])
    u = np.array([1.0])
    v = np.array([0.0])
    src_proj = ccrs.PlateCarree()
    target_proj = ccrs.Stereographic(central_latitude=90, central_longitude=0)
    ut, vt = target_proj.transform_vectors(src_proj, rlon, rlat, u, v)
    assert_array_almost_equal(ut, np.array([0]), decimal=2)
    assert_array_almost_equal(vt, np.array([-1]), decimal=2)