from unittest.mock import Mock
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import (
def test_LatitudeFormatter_small_numbers():
    formatter = LatitudeFormatter(number_format='.7f', dms=False)
    p = ccrs.PlateCarree()
    formatter.set_axis(Mock(axes=Mock(GeoAxes, projection=p)))
    test_ticks = [40.127515, 40.1275152, 40.1275154]
    result = [formatter(tick) for tick in test_ticks]
    expected = ['40.1275150°N', '40.1275152°N', '40.1275154°N']
    assert result == expected