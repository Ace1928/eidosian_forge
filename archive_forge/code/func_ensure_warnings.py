import functools
import warnings
from collections.abc import Hashable
from typing import Union
import numpy as np
import pandas as pd
from xarray.core import duck_array_ops, formatting, utils
from xarray.core.coordinates import Coordinates
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.indexes import Index, PandasIndex, PandasMultiIndex, default_indexes
from xarray.core.variable import IndexVariable, Variable
def ensure_warnings(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        __tracebackhide__ = True
        with warnings.catch_warnings():
            warnings.filters = [f for f in warnings.filters if f[0] != 'error']
            return func(*args, **kwargs)
    return wrapper