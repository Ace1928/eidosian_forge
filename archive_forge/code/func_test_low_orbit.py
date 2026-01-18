from numpy.testing import assert_almost_equal
import cartopy.crs as ccrs
from .helpers import check_proj_params
def test_low_orbit(self):
    geos = self.test_class(satellite_height=700000)
    other_args = {'ellps=WGS84', 'h=700000', 'lat_0=0.0', 'lon_0=0.0', 'units=m', 'x_0=0', 'y_0=0'}
    self.adjust_expected_params(other_args)
    check_proj_params(self.expected_proj_name, geos, other_args)
    assert_almost_equal(geos.boundary.bounds, (-785616.1189, -783815.6629, 785616.1189, 783815.6629), decimal=4)
    assert_almost_equal(geos.boundary.coords[7], (750051.0347, -305714.8243), decimal=4)