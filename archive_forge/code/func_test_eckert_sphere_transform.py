import numpy as np
from numpy.testing import assert_almost_equal
import pytest
import cartopy.crs as ccrs
from .helpers import check_proj_params
@pytest.mark.parametrize('name, proj, lim, expected', [pytest.param('eck4', ccrs.EckertIV, 2.65300085, [0.187527, -0.951921], id='EckertIV'), pytest.param('eck6', ccrs.EckertVI, 2.77096497, [0.1693623, -0.9570223], id='EckertVI')])
def test_eckert_sphere_transform(name, proj, lim, expected):
    globe = ccrs.Globe(semimajor_axis=1.0, ellipse=None)
    eck = proj(central_longitude=-90.0, globe=globe)
    geodetic = eck.as_geodetic()
    other_args = {'a=1.0', 'lon_0=-90.0'}
    check_proj_params(name, eck, other_args)
    assert_almost_equal(eck.x_limits, [-lim, lim], decimal=2)
    assert_almost_equal(eck.y_limits, [-lim / 2, lim / 2])
    result = eck.transform_point(-75.0, -50.0, geodetic)
    assert_almost_equal(result, expected)
    inverse_result = geodetic.transform_point(result[0], result[1], eck)
    assert_almost_equal(inverse_result, [-75.0, -50.0])