from __future__ import annotations
from io import BytesIO
import numpy as np
import xarray as xr
import dask.array as da
import PIL
import pytest
import datashader.transfer_functions as tf
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr, assert_image_close
def create_dask_array_np(*args, **kwargs):
    """Create a dask array wrapping around a numpy array."""
    return da.from_array(np.array(*args, **kwargs))