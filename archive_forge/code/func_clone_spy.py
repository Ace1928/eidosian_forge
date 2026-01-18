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
@pytest.fixture
def clone_spy():
    spy = Spy()
    _clone = Interactive._clone

    def clone_bis(inst, *args, **kwargs):
        cloned = _clone(inst, *args, **kwargs)
        spy.register_call(args, kwargs, depth=cloned._depth)
        return cloned
    Interactive._clone = clone_bis
    yield spy
    Interactive._clone = _clone