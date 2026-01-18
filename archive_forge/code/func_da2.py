import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
@pytest.fixture(scope='module')
def da2():
    return xr.DataArray(data=np.arange(27).reshape((3, 3, 3)), coords={'y': [0, 1, 2], 'x': [0, 1, 2]}, dims=['y', 'x', 'other'], name='test2')