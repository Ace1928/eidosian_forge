from unittest.mock import Mock
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import (
@pytest.mark.parametrize('test_ticks,expected', [pytest.param([-3.75, -3.5], ['3°45′S', '3°30′S'], id='minutes_no_hide')])
def test_LatitudeFormatter_minutes_seconds(test_ticks, expected):
    formatter = LatitudeFormatter(dms=True, auto_hide=True)
    formatter.set_locs(test_ticks)
    result = [formatter(tick) for tick in test_ticks]
    assert result == expected