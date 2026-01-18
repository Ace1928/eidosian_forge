from numpy.testing import assert_array_almost_equal
import pyproj
import pytest
import cartopy.crs as ccrs
from .helpers import check_proj_params
def test_lambert_moon():
    moon = ccrs.Globe(ellipse=None, semimajor_axis=1737400, semiminor_axis=1737400)
    crs = ccrs.LambertConformal(globe=moon)
    other_args = {'a=1737400', 'b=1737400', 'lat_0=39.0', 'lat_1=33', 'lat_2=45', 'lon_0=-96.0', 'x_0=0.0', 'y_0=0.0'}
    check_proj_params('lcc', crs, other_args)