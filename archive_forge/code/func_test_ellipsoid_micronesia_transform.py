import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import cartopy.crs as ccrs
from .helpers import check_proj_params
def test_ellipsoid_micronesia_transform(self):
    globe = ccrs.Globe(ellipse=None, semimajor_axis=6378206.4, flattening=1 - np.sqrt(1 - 0.00676866))
    lat_0 = 15 + (11 + 5.683 / 60) / 60
    lon_0 = 145 + (44 + 29.972 / 60) / 60
    aeqd = ccrs.AzimuthalEquidistant(central_latitude=lat_0, central_longitude=lon_0, false_easting=28657.52, false_northing=67199.99, globe=globe)
    geodetic = aeqd.as_geodetic()
    other_args = {'a=6378206.4', 'f=0.003390076308689371', 'lon_0=145.7416588888889', 'lat_0=15.18491194444444', 'x_0=28657.52', 'y_0=67199.99000000001'}
    check_proj_params('aeqd', aeqd, other_args)
    assert_almost_equal(np.array(aeqd.x_limits), [-20009068.8493194, 20066383.8893194], decimal=6)
    assert_almost_equal(np.array(aeqd.y_limits), [-19902596.95787477, 20036996.93787477], decimal=6)
    pt_lat = 15 + (14 + 47.493 / 60) / 60
    pt_lon = 145 + (47 + 34.908 / 60) / 60
    result = aeqd.transform_point(pt_lon, pt_lat, geodetic)
    assert_array_almost_equal(result, [34176.2, 74017.88], decimal=2)