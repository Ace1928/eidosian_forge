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
def test_coordinates_encoding(self) -> None:

    def equals_latlon(obj):
        return obj == 'lat lon' or obj == 'lon lat'
    original = Dataset({'temp': ('x', [0, 1]), 'precip': ('x', [0, -1])}, {'lat': ('x', [2, 3]), 'lon': ('x', [4, 5])})
    with self.roundtrip(original) as actual:
        assert_identical(actual, original)
    with create_tmp_file() as tmp_file:
        original.to_netcdf(tmp_file)
        with open_dataset(tmp_file, decode_coords=False) as ds:
            assert equals_latlon(ds['temp'].attrs['coordinates'])
            assert equals_latlon(ds['precip'].attrs['coordinates'])
            assert 'coordinates' not in ds.attrs
            assert 'coordinates' not in ds['lat'].attrs
            assert 'coordinates' not in ds['lon'].attrs
    modified = original.drop_vars(['temp', 'precip'])
    with self.roundtrip(modified) as actual:
        assert_identical(actual, modified)
    with create_tmp_file() as tmp_file:
        modified.to_netcdf(tmp_file)
        with open_dataset(tmp_file, decode_coords=False) as ds:
            assert equals_latlon(ds.attrs['coordinates'])
            assert 'coordinates' not in ds['lat'].attrs
            assert 'coordinates' not in ds['lon'].attrs
    original['temp'].encoding['coordinates'] = 'lat'
    with self.roundtrip(original) as actual:
        assert_identical(actual, original)
    original['precip'].encoding['coordinates'] = 'lat'
    with create_tmp_file() as tmp_file:
        original.to_netcdf(tmp_file)
        with open_dataset(tmp_file, decode_coords=True) as ds:
            assert 'lon' not in ds['temp'].encoding['coordinates']
            assert 'lon' not in ds['precip'].encoding['coordinates']
            assert 'coordinates' not in ds['lat'].encoding
            assert 'coordinates' not in ds['lon'].encoding