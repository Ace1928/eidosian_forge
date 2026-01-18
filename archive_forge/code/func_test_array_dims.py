from functools import reduce
import operator
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from cartopy import config
import cartopy.crs as ccrs
from cartopy.tests.conftest import _HAS_PYKDTREE_OR_SCIPY
import cartopy.img_transform as im_trans
def test_array_dims(self):
    source_nx = 100
    source_ny = 100
    source_x = np.linspace(-180.0, 180.0, source_nx).astype(np.float64)
    source_y = np.linspace(-90, 90.0, source_ny).astype(np.float64)
    source_x, source_y = np.meshgrid(source_x, source_y)
    data = np.arange(source_nx * source_ny, dtype=np.int32).reshape(source_ny, source_nx)
    source_cs = ccrs.Geodetic()
    target_nx = 23
    target_ny = 45
    target_proj = ccrs.PlateCarree()
    target_x, target_y, extent = im_trans.mesh_projection(target_proj, target_nx, target_ny)
    new_array = im_trans.regrid(data, source_x, source_y, source_cs, target_proj, target_x, target_y)
    assert new_array.shape == target_x.shape
    assert new_array.shape == target_y.shape
    assert new_array.shape == (target_ny, target_nx)