import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from packaging.version import Version
from matplotlib import cm
from matplotlib import colors
from branca.colormap import StepColormap
def test_custom_markers(self):
    m = self.cities.explore(marker_type='marker', marker_kwds={'icon': folium.Icon(icon='star')})
    assert ',"icon":"star",' in self._fetch_map_string(m)
    m = self.cities.explore(marker_type='circle', marker_kwds={'fill_color': 'red'})
    assert ',"fillColor":"red",' in self._fetch_map_string(m)
    m = self.cities.explore(marker_type=folium.Circle(radius=4, fill_color='orange', fill_opacity=0.4, color='black', weight=1))
    assert ',"color":"black",' in self._fetch_map_string(m)
    m = self.cities.explore(marker_type='circle_marker', marker_kwds={'radius': 10})
    assert ',"radius":10,' in self._fetch_map_string(m)
    with pytest.raises(ValueError, match="Only 'marker', 'circle', and 'circle_marker' are supported"):
        self.cities.explore(marker_type='dummy')