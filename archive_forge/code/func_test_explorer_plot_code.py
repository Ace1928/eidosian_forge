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
def test_explorer_plot_code():
    explorer = hvplot.explorer(df)
    explorer.param.update(kind='scatter', x='bill_length_mm', y_multi=['bill_depth_mm'], by=['species'])
    hvplot_code = explorer.plot_code()
    assert hvplot_code == "df.hvplot(\n    by=['species'],\n    kind='scatter',\n    x='bill_length_mm',\n    y=['bill_depth_mm'],\n    legend='bottom_right',\n    widget_location='bottom',\n)"
    hvplot_code = explorer.plot_code(var_name='othername')
    assert hvplot_code == "othername.hvplot(\n    by=['species'],\n    kind='scatter',\n    x='bill_length_mm',\n    y=['bill_depth_mm'],\n    legend='bottom_right',\n    widget_location='bottom',\n)"