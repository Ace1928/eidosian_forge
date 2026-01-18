from __future__ import annotations
import contextlib
import gzip
import itertools
import math
import os.path
import pickle
import platform
import re
import shutil
import sys
import tempfile
import uuid
import warnings
from collections.abc import Generator, Iterator, Mapping
from contextlib import ExitStack
from io import BytesIO
from os import listdir
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, cast
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
from pandas.errors import OutOfBoundsDatetime
import xarray as xr
from xarray import (
from xarray.backends.common import robust_getitem
from xarray.backends.h5netcdf_ import H5netcdfBackendEntrypoint
from xarray.backends.netcdf3 import _nc3_dtype_coercions
from xarray.backends.netCDF4_ import (
from xarray.backends.pydap_ import PydapDataStore
from xarray.backends.scipy_ import ScipyBackendEntrypoint
from xarray.coding.cftime_offsets import cftime_range
from xarray.coding.strings import check_vlen_dtype, create_vlen_dtype
from xarray.coding.variables import SerializationWarning
from xarray.conventions import encode_dataset_coordinates
from xarray.core import indexing
from xarray.core.options import set_options
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
from xarray.tests.test_coding_times import (
from xarray.tests.test_dataset import (
@requires_zarr
@requires_dask
@pytest.mark.parametrize('chunks', ['auto', -1, {}, {'x': 'auto'}, {'x': -1}, {'x': 'auto', 'y': -1}])
@pytest.mark.filterwarnings('ignore:The specified chunks separate')
def test_chunking_consintency(chunks, tmp_path: Path) -> None:
    encoded_chunks: dict[str, Any] = {}
    dask_arr = da.from_array(np.ones((500, 500), dtype='float64'), chunks=encoded_chunks)
    ds = xr.Dataset({'test': xr.DataArray(dask_arr, dims=('x', 'y'))})
    ds['test'].encoding['chunks'] = encoded_chunks
    ds.to_zarr(tmp_path / 'test.zarr')
    ds.to_netcdf(tmp_path / 'test.nc')
    with dask.config.set({'array.chunk-size': '1MiB'}):
        expected = ds.chunk(chunks)
        with xr.open_dataset(tmp_path / 'test.zarr', engine='zarr', chunks=chunks) as actual:
            xr.testing.assert_chunks_equal(actual, expected)
        with xr.open_dataset(tmp_path / 'test.nc', chunks=chunks) as actual:
            xr.testing.assert_chunks_equal(actual, expected)