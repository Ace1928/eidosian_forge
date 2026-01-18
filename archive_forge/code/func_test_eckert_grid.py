import numpy as np
from numpy.testing import assert_almost_equal
import pytest
import cartopy.crs as ccrs
from .helpers import check_proj_params
@pytest.mark.parametrize('name, proj, radius, expected_x, expected_y', [pytest.param('eck4', ccrs.EckertIV, 0.75386, np.array([0.5, 0.55613, 0.6082, 0.65656, 0.70141, 0.74291, 0.78117, 0.81625, 0.84822, 0.87709, 0.90291, 0.92567, 0.94539, 0.96208, 0.97573, 0.98635, 0.99393, 0.99848, 1.0]), np.array([1.0, 0.99368, 0.9763, 0.94971, 0.91528, 0.87406, 0.82691, 0.77455, 0.71762, 0.65666, 0.59217, 0.52462, 0.45443, 0.38202, 0.30779, 0.2321, 0.15533, 0.07784, 0.0]), id='EckertIV'), pytest.param('eck6', ccrs.EckertVI, 0.72177, np.array([0.5, 0.50487, 0.51916, 0.54198, 0.57205, 0.60782, 0.64767, 0.69004, 0.73344, 0.77655, 0.81817, 0.85724, 0.89288, 0.9243, 0.95087, 0.97207, 0.98749, 0.99686, 1.0]), np.array([1.0, 0.9938, 0.9756, 0.94648, 0.90794, 0.86164, 0.80913, 0.7518, 0.69075, 0.62689, 0.5609, 0.49332, 0.42454, 0.35488, 0.28457, 0.21379, 0.14269, 0.0714, 0.0]), id='EckertVI')])
def test_eckert_grid(name, proj, radius, expected_x, expected_y):
    globe = ccrs.Globe(semimajor_axis=radius, ellipse=None)
    eck = proj(globe=globe)
    geodetic = eck.as_geodetic()
    other_args = {f'a={radius}', 'lon_0=0'}
    check_proj_params(name, eck, other_args)
    assert_almost_equal(eck.x_limits, [-2, 2], decimal=5)
    assert_almost_equal(eck.y_limits, [-1, 1], decimal=5)
    lats = np.arange(0, 91, 5)[::-1]
    lons = np.full_like(lats, 90)
    result = eck.transform_points(geodetic, lons, lats)
    assert_almost_equal(result[:, 0], expected_x, decimal=5)
    assert_almost_equal(result[:, 1], expected_y, decimal=5)