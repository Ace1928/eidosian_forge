from unittest.mock import Mock
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import (
@pytest.mark.parametrize('central_longitude, kwargs, expected', [(0, {'dateline_direction_label': True}, ['180°W', '120°W', '60°W', '0°', '60°E', '120°E', '180°E']), (180, {'zero_direction_label': True}, ['0°E', '60°E', '120°E', '180°', '120°W', '60°W', '0°W']), (120, {}, ['60°W', '0°', '60°E', '120°E', '180°', '120°W', '60°W'])])
def test_LongitudeFormatter_central_longitude(central_longitude, kwargs, expected):
    formatter = LongitudeFormatter(**kwargs)
    p = ccrs.PlateCarree(central_longitude=central_longitude)
    formatter.set_axis(Mock(axes=Mock(GeoAxes, projection=p)))
    test_ticks = [-180, -120, -60, 0, 60, 120, 180]
    result = [formatter(tick) for tick in test_ticks]
    assert result == expected