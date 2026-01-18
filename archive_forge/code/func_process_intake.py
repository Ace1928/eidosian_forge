import sys
from collections.abc import Hashable
from functools import wraps
from packaging.version import Version
from types import FunctionType
import bokeh
import numpy as np
import pandas as pd
import param
import holoviews as hv
def process_intake(data, use_dask):
    if data.container not in ('dataframe', 'xarray'):
        raise NotImplementedError('Plotting interface currently only supports DataSource objects declaring a dataframe or xarray container.')
    if use_dask:
        data = data.to_dask()
    else:
        data = data.read()
    return data