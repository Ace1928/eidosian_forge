from unittest.mock import Mock
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import (
@pytest.mark.parametrize('cls,vmin,vmax,expected', [pytest.param(LongitudeLocator, -180, 180, [-180, -120, -60, 0, 60, 120, 180], id='lon_large'), pytest.param(LatitudeLocator, -180, 180, [-90, -60, -30, 0, 30, 60, 90], id='lat_large'), pytest.param(LongitudeLocator, -10, 0, [-10.5, -9, -7.5, -6, -4.5, -3, -1.5, 0], id='lon_medium'), pytest.param(LongitudeLocator, -1, 0, np.array([-60, -50, -40, -30, -20, -10, 0]) / 60, id='lon_small'), pytest.param(LongitudeLocator, 0, 2 * ONE_MIN, np.array([0, 18, 36, 54, 72, 90, 108, 126]) / 3600, id='lon_tiny')])
def test_LongitudeLocator(cls, vmin, vmax, expected):
    locator = cls(dms=True)
    result = locator.tick_values(vmin, vmax)
    np.testing.assert_allclose(result, expected)