from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
def assert_eq_ndarray(data, b, close=False):
    """Assert that two ndarrays are equal, handling the possibility that the
    ndarrays are of different types"""
    if cupy:
        if isinstance(data, cupy.ndarray):
            data = cupy.asnumpy(data)
        if isinstance(b, cupy.ndarray):
            b = cupy.asnumpy(b)
    if close:
        np.testing.assert_array_almost_equal(data, b, decimal=5)
    else:
        np.testing.assert_equal(data, b)