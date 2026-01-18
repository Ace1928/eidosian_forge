import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from packaging.version import Version
from matplotlib import cm
from matplotlib import colors
from branca.colormap import StepColormap
def test_map_settings_custom(self):
    """Check custom map settings"""
    m = self.nybb.explore(zoom_control=False, width=200, height=200)
    assert m.location == [pytest.approx(40.70582377450201, rel=1e-06), pytest.approx(-73.9778006856748, rel=1e-06)]
    assert m.options['zoom'] == 10
    assert m.options['zoomControl'] is False
    assert m.height == (200.0, 'px')
    assert m.width == (200.0, 'px')
    m = self.nybb.explore(zoom_control=False, width=200, height=200, tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}', attr='Google')
    out_str = self._fetch_map_string(m)
    s = '"https://mt1.google.com/vt/lyrs=m\\u0026x={x}\\u0026y={y}\\u0026z={z}"'
    assert s in out_str
    assert '"attribution":"Google"' in out_str
    m = self.nybb.explore(location=(40, 5))
    assert m.location == [40, 5]
    assert m.options['zoom'] == 10
    m = self.nybb.explore(zoom_start=8)
    assert m.location == [pytest.approx(40.70582377450201, rel=1e-06), pytest.approx(-73.9778006856748, rel=1e-06)]
    assert m.options['zoom'] == 8
    m = self.nybb.explore(location=(40, 5), zoom_start=8)
    assert m.location == [40, 5]
    assert m.options['zoom'] == 8