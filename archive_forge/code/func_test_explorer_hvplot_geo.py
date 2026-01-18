import re
from textwrap import dedent
import holoviews as hv
import pandas as pd
import hvplot.pandas
import hvplot.xarray
import xarray as xr
import pytest
from bokeh.sampledata import penguins
from hvplot.ui import hvDataFrameExplorer, hvGridExplorer
def test_explorer_hvplot_geo():
    pytest.importorskip('geoviews')
    df = pd.DataFrame({'x': [-9796115.18980811], 'y': [4838471.398061159]})
    explorer = hvplot.explorer(df, x='x', geo=True, kind='points')
    assert explorer.geographic.geo
    assert explorer.geographic.global_extent
    assert explorer.geographic.features == ['coastline']
    assert explorer.geographic.crs == 'GOOGLE_MERCATOR'
    assert explorer.geographic.projection == 'GOOGLE_MERCATOR'