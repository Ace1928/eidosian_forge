from numpy.testing import assert_array_almost_equal
import pyproj
import pytest
import cartopy.crs as ccrs
from .helpers import check_proj_params
def test_no_parallel(self):
    with pytest.raises(ValueError, match='1 or 2 standard parallels'):
        ccrs.LambertConformal(standard_parallels=[])