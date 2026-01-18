import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from packaging.version import Version
from matplotlib import cm
from matplotlib import colors
from branca.colormap import StepColormap
def test_given_m(self):
    """Check that geometry is mapped onto a given folium.Map"""
    m = folium.Map()
    self.nybb.explore(m=m, tooltip=False, highlight=False)
    out_str = self._fetch_map_string(m)
    assert out_str.count('BoroCode') == 5
    assert m.options['zoom'] == 1