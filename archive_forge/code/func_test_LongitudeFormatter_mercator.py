from unittest.mock import Mock
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import (
def test_LongitudeFormatter_mercator():
    formatter = LongitudeFormatter(dateline_direction_label=True)
    p = ccrs.Mercator()
    formatter.set_axis(Mock(axes=Mock(GeoAxes, projection=p)))
    test_ticks = [-20037508.342783064, -13358338.895188706, -6679169.447594353, 0.0, 6679169.447594353, 13358338.895188706, 20037508.342783064]
    result = [formatter(tick) for tick in test_ticks]
    expected = ['180°W', '120°W', '60°W', '0°', '60°E', '120°E', '180°E']
    assert result == expected