from unittest.mock import Mock
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import (
@pytest.mark.parametrize('cls', [LatitudeFormatter, LongitudeFormatter])
def test_formatter_bad_projection(cls):
    formatter = cls()
    match = 'This formatter cannot be used with non-rectangular projections\\.'
    with pytest.raises(TypeError, match=match):
        formatter.set_axis(Mock(axes=Mock(GeoAxes, projection=ccrs.Orthographic())))