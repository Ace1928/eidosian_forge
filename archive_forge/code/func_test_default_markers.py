import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from packaging.version import Version
from matplotlib import cm
from matplotlib import colors
from branca.colormap import StepColormap
def test_default_markers(self):
    m = self.cities.explore()
    strings = ['"radius":2', '"fill":true', 'CircleMarker(latlng,opts)']
    out_str = self._fetch_map_string(m)
    for s in strings:
        assert s in out_str
    m = self.cities.explore(marker_kwds={'radius': 5, 'fill': False})
    strings = ['"radius":5', '"fill":false', 'CircleMarker(latlng,opts)']
    out_str = self._fetch_map_string(m)
    for s in strings:
        assert s in out_str