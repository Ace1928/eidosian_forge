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
def test_interactive_pandas_series_plot_kind_attr(series, clone_spy):
    si = Interactive(series)
    si = si.plot.line()
    assert isinstance(si, Interactive)
    assert isinstance(si._current, matplotlib.axes.Axes)
    assert si._obj is series
    assert si._depth == 4
    assert si._method is None
    assert clone_spy.count == 4
    assert clone_spy.calls[0].depth == 1
    assert not clone_spy.calls[0].args
    assert clone_spy.calls[0].kwargs == {'copy': True}
    assert clone_spy.calls[1].depth == 2
    assert len(clone_spy.calls[1].args) == 1
    assert len(clone_spy.calls[1].kwargs) == 1
    assert 'inherit_kwargs' in clone_spy.calls[1].kwargs
    assert 'ax' in clone_spy.calls[1].kwargs['inherit_kwargs']
    assert clone_spy.calls[2].depth == 3
    assert not clone_spy.calls[2].args
    assert clone_spy.calls[2].kwargs == {'copy': True}
    assert clone_spy.calls[3].depth == 4
    assert len(clone_spy.calls[3].args) == 1
    assert clone_spy.calls[3].kwargs == {'plot': False}
    assert not si._dmap
    assert isinstance(si._fig, matplotlib.figure.Figure)
    si.output()