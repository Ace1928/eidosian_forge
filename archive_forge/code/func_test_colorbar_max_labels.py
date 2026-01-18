import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from packaging.version import Version
from matplotlib import cm
from matplotlib import colors
from branca.colormap import StepColormap
@pytest.mark.skipif(not BRANCA_05, reason='requires branca >= 0.5.0')
def test_colorbar_max_labels(self):
    import re
    m = self.world.explore('pop_est', legend_kwds={'max_labels': 3})
    out_str = self._fetch_map_string(m)
    tick_str = re.search("tickValues\\(\\[[\\',\\,\\.,0-9]*\\]\\)", out_str).group(0)
    assert tick_str.replace(",''", '') == 'tickValues([140.0,471386328.07843137,942772516.1568627])'
    m = self.world.explore('pop_est', scheme='headtailbreaks', legend_kwds={'max_labels': 3})
    out_str = self._fetch_map_string(m)
    assert "tickValues([140.0,'',184117213.1818182,'',1382066377.0,''])" in out_str
    m = self.world.explore('pop_est', legend_kwds={'max_labels': 3}, cmap='tab10')
    out_str = self._fetch_map_string(m)
    tick_str = re.search("tickValues\\(\\[[\\',\\,\\.,0-9]*\\]\\)", out_str).group(0)
    assert tick_str == "tickValues([140.0,'','','',559086084.0,'','','',1118172028.0,'','',''])"