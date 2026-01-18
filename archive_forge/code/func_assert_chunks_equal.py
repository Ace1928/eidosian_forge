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
def assert_chunks_equal(a, b):
    """
    Assert that chunksizes along chunked dimensions are equal.

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        The first object to compare.
    b : xarray.Dataset or xarray.DataArray
        The second object to compare.
    """
    if isinstance(a, DataArray) != isinstance(b, DataArray):
        raise TypeError('a and b have mismatched types')
    left = a.unify_chunks()
    right = b.unify_chunks()
    assert left.chunks == right.chunks