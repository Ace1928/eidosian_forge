from packaging.version import Version
import holoviews as hv
import hvplot.pandas  # noqa
import hvplot.xarray  # noqa
import matplotlib
import numpy as np
import pandas as pd
import panel as pn
import pytest
import xarray as xr
from holoviews.util.transform import dim
from hvplot import bind
from hvplot.interactive import Interactive
from hvplot.tests.util import makeDataFrame, makeMixedDataFrame
from hvplot.xarray import XArrayInteractive
from hvplot.util import bokeh3, param2
def test_interactive_nested_widgets():
    df = makeDataFrame()
    w = pn.widgets.RadioButtonGroup(value='A', options=list('ABC'))
    idf = Interactive(df)
    pipeline = idf.groupby(['D', w]).mean()
    ioutput = pipeline.panel().object().object
    iw = pipeline.widgets()
    output = df.groupby(['D', 'A']).mean()
    pd.testing.assert_frame_equal(ioutput, output)
    assert len(iw) == 1
    assert iw[0] == w