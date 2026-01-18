import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import cartopy.crs as ccrs
from .helpers import check_proj_params
def test_ellipsoid_polar_transform(self):
    globe = ccrs.Globe(ellipse=None, semimajor_axis=6378388.0, flattening=1 - np.sqrt(1 - 0.00672267))
    aeqd = ccrs.AzimuthalEquidistant(central_latitude=90.0, central_longitude=-100.0, globe=globe)
    geodetic = aeqd.as_geodetic()
    other_args = {'a=6378388.0', 'f=0.003367003355798981', 'lon_0=-100.0', 'lat_0=90.0', 'x_0=0.0', 'y_0=0.0'}
    check_proj_params('aeqd', aeqd, other_args)
    assert_almost_equal(np.array(aeqd.x_limits), [-20038296.88254529, 20038296.88254529], decimal=6)
    assert_almost_equal(np.array(aeqd.y_limits), [-19970827.86969727, 19970827.86969727], decimal=6)
    result = aeqd.transform_point(5.0, 80.0, geodetic)
    assert_array_almost_equal(result, [1078828.3, 289071.2], decimal=1)