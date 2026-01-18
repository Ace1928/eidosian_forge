import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from packaging.version import Version
from matplotlib import cm
from matplotlib import colors
from branca.colormap import StepColormap
def test_mapclassify_categorical_legend(self):
    m = self.missing.explore(column='pop_est', legend=True, scheme='naturalbreaks', missing_kwds={'color': 'red', 'label': 'missing'}, legend_kwds={'colorbar': False, 'interval': True})
    out_str = self._fetch_map_string(m)
    strings = ['[140.00,21803000.00]', '(21803000.00,66834405.00]', '(66834405.00,163046161.00]', '(163046161.00,328239523.00]', '(328239523.00,1397715000.00]', 'missing']
    for s in strings:
        assert s in out_str
    m = self.missing.explore(column='pop_est', legend=True, scheme='naturalbreaks', missing_kwds={'color': 'red', 'label': 'missing'}, legend_kwds={'colorbar': False, 'interval': False})
    out_str = self._fetch_map_string(m)
    strings = ['>140.00,21803000.00', '>21803000.00,66834405.00', '>66834405.00,163046161.00', '>163046161.00,328239523.00', '>328239523.00,1397715000.00', 'missing']
    for s in strings:
        assert s in out_str
    m = self.world.explore(column='pop_est', legend=True, scheme='naturalbreaks', k=5, legend_kwds={'colorbar': False, 'labels': ['s', 'm', 'l', 'xl', 'xxl']})
    out_str = self._fetch_map_string(m)
    strings = ['>s<', '>m<', '>l<', '>xl<', '>xxl<']
    for s in strings:
        assert s in out_str
    m = self.missing.explore(column='pop_est', legend=True, scheme='naturalbreaks', missing_kwds={'color': 'red', 'label': 'missing'}, legend_kwds={'colorbar': False, 'fmt': '{:.0f}'})
    out_str = self._fetch_map_string(m)
    strings = ['>140,21803000', '>21803000,66834405', '>66834405,163046161', '>163046161,328239523', '>328239523,1397715000', 'missing']
    for s in strings:
        assert s in out_str