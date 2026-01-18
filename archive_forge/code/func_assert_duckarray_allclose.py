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
def assert_duckarray_allclose(actual, desired, rtol=1e-07, atol=0, err_msg='', verbose=True):
    """Like `np.testing.assert_allclose`, but for duckarrays."""
    __tracebackhide__ = True
    allclose = duck_array_ops.allclose_or_equiv(actual, desired, rtol=rtol, atol=atol)
    assert allclose, _format_message(actual, desired, err_msg=err_msg, verbose=verbose)