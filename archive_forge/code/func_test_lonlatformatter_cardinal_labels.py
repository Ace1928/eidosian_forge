from unittest.mock import Mock
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import (
def test_lonlatformatter_cardinal_labels():
    xticker = LongitudeFormatter(cardinal_labels={'west': 'O'})
    yticker = LatitudeFormatter(cardinal_labels={'south': 'South'})
    assert xticker(-10) == '10°O'
    assert yticker(-10) == '10°South'