import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from packaging.version import Version
from matplotlib import cm
from matplotlib import colors
from branca.colormap import StepColormap
def test_missing_vals(self):
    m = self.missing.explore('continent')
    assert '"fillColor":null' in self._fetch_map_string(m)
    m = self.missing.explore('pop_est')
    assert '"fillColor":null' in self._fetch_map_string(m)
    m = self.missing.explore('pop_est', missing_kwds={'color': 'red'})
    assert '"fillColor":"red"' in self._fetch_map_string(m)
    m = self.missing.explore('continent', missing_kwds={'color': 'red'})
    assert '"fillColor":"red"' in self._fetch_map_string(m)