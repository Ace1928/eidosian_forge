from unittest.mock import Mock
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import (
def test_LongitudeFormatter_direction_label():
    formatter = LongitudeFormatter(direction_label=False, dateline_direction_label=True, zero_direction_label=True)
    p = ccrs.PlateCarree()
    formatter.set_axis(Mock(axes=Mock(GeoAxes, projection=p)))
    test_ticks = [-180, -120, -60, 0, 60, 120, 180]
    result = [formatter(tick) for tick in test_ticks]
    expected = ['-180°', '-120°', '-60°', '0°', '60°', '120°', '180°']
    assert result == expected