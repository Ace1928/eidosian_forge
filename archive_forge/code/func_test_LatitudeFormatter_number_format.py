from unittest.mock import Mock
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import (
def test_LatitudeFormatter_number_format():
    formatter = LatitudeFormatter(number_format='.2f', dms=False)
    p = ccrs.PlateCarree()
    formatter.set_axis(Mock(axes=Mock(GeoAxes, projection=p)))
    test_ticks = [-90, -60, -30, 0, 30, 60, 90]
    result = [formatter(tick) for tick in test_ticks]
    expected = ['90.00°S', '60.00°S', '30.00°S', '0.00°', '30.00°N', '60.00°N', '90.00°N']
    assert result == expected