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
@ensure_warnings
def assert_duckarray_equal(x, y, err_msg='', verbose=True):
    """Like `np.testing.assert_array_equal`, but for duckarrays"""
    __tracebackhide__ = True
    if not utils.is_duck_array(x) and (not utils.is_scalar(x)):
        x = np.asarray(x)
    if not utils.is_duck_array(y) and (not utils.is_scalar(y)):
        y = np.asarray(y)
    if utils.is_duck_array(x) and utils.is_scalar(y) or (utils.is_scalar(x) and utils.is_duck_array(y)):
        equiv = (x == y).all()
    else:
        equiv = duck_array_ops.array_equiv(x, y)
    assert equiv, _format_message(x, y, err_msg=err_msg, verbose=verbose)