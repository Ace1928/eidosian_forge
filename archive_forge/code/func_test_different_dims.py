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
def test_different_dims(self):
    source_nx = 100
    source_ny = 100
    source_x = np.linspace(-180.0, 180.0, source_nx).astype(np.float64)
    source_y = np.linspace(-90, 90.0, source_ny).astype(np.float64)
    source_x, source_y = np.meshgrid(source_x, source_y)
    data = np.arange(source_nx * source_ny, dtype=np.int32).reshape(source_ny, source_nx)
    source_cs = ccrs.Geodetic()
    target_x_shape = (23, 45)
    target_y_shape = (23, 44)
    target_x = np.arange(reduce(operator.mul, target_x_shape), dtype=np.float64).reshape(target_x_shape)
    target_y = np.arange(reduce(operator.mul, target_y_shape), dtype=np.float64).reshape(target_y_shape)
    target_proj = ccrs.PlateCarree()
    with pytest.raises(ValueError):
        im_trans.regrid(data, source_x, source_y, source_cs, target_proj, target_x, target_y)