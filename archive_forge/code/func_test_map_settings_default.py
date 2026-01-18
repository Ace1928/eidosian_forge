import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from packaging.version import Version
from matplotlib import cm
from matplotlib import colors
from branca.colormap import StepColormap
def test_map_settings_default(self):
    """Check default map settings"""
    m = self.world.explore()
    assert m.location == [pytest.approx(-3.1774349999999956, rel=1e-06), pytest.approx(2.842170943040401e-14, rel=1e-06)]
    assert m.options['zoom'] == 10
    assert m.options['zoomControl'] is True
    assert m.position == 'relative'
    assert m.height == (100.0, '%')
    assert m.width == (100.0, '%')
    assert m.left == (0, '%')
    assert m.top == (0, '%')
    assert m.global_switches.no_touch is False
    assert m.global_switches.disable_3d is False
    assert 'openstreetmap' in m.to_dict()['children'].keys()