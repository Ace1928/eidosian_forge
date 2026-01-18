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
def test_explorer_settings():
    explorer = hvplot.explorer(df)
    explorer.param.update(kind='scatter', x='bill_length_mm', y_multi=['bill_depth_mm'], by=['species'])
    settings = explorer.settings()
    assert settings == dict(by=['species'], kind='scatter', x='bill_length_mm', y=['bill_depth_mm'])