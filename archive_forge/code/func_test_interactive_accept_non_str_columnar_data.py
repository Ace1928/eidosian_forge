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
def test_interactive_accept_non_str_columnar_data():
    df = pd.DataFrame(np.random.random((10, 2)))
    assert all((not isinstance(col, str) for col in df.columns))
    dfi = Interactive(df)
    w = pn.widgets.FloatSlider(start=0, end=1, step=0.05)
    dfi = dfi['1'] + w.param.value
    w.value = 0.5
    pytest.approx(dfi.eval().sum(), (df[1] + 0.5).sum())