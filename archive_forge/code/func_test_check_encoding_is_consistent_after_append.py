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
def test_check_encoding_is_consistent_after_append(self) -> None:
    ds, ds_to_append, _ = create_append_test_data()
    with self.create_zarr_target() as store_target:
        import zarr
        compressor = zarr.Blosc()
        encoding = {'da': {'compressor': compressor}}
        ds.to_zarr(store_target, mode='w', encoding=encoding, **self.version_kwargs)
        ds_to_append.to_zarr(store_target, append_dim='time', **self.version_kwargs)
        actual_ds = xr.open_dataset(store_target, engine='zarr', **self.version_kwargs)
        actual_encoding = actual_ds['da'].encoding['compressor']
        assert actual_encoding.get_config() == compressor.get_config()
        assert_identical(xr.open_dataset(store_target, engine='zarr', **self.version_kwargs).compute(), xr.concat([ds, ds_to_append], dim='time'))