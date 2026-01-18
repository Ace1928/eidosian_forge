import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
import cartopy.crs as ccrs
from cartopy.tests.conftest import requires_scipy
def test_contour_doesnt_shrink():
    xglobal = np.linspace(-180, 180)
    yglobal = np.linspace(-90, 90)
    xsmall = np.linspace(-30, 30)
    ysmall = np.linspace(-30, 30)
    data = np.hypot(*np.meshgrid(xglobal, yglobal))
    proj = ccrs.PlateCarree()
    ax = plt.axes(projection=proj)
    ax.contourf(xglobal, yglobal, data)
    expected = np.array([xglobal[0], xglobal[-1], yglobal[0], yglobal[-1]])
    assert_array_almost_equal(ax.get_extent(), expected)
    ax.contour(xsmall, ysmall, data)
    assert_array_almost_equal(ax.get_extent(), expected)
    ax.contourf(xsmall, ysmall, data)
    assert_array_almost_equal(ax.get_extent(), expected)