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
@requires_dask
@pytest.mark.filterwarnings('ignore:The specified chunks separate:UserWarning')
def test_manual_chunk(self) -> None:
    original = create_test_data().chunk({'dim1': 3, 'dim2': 4, 'dim3': 3})
    open_kwargs: dict[str, Any] = {'chunks': None}
    with self.roundtrip(original, open_kwargs=open_kwargs) as actual:
        for k, v in actual.variables.items():
            assert v._in_memory == (k in actual.dims)
            assert v.chunks is None
    for i in range(2, 6):
        rechunked = original.chunk(chunks=i)
        open_kwargs = {'chunks': i}
        with self.roundtrip(original, open_kwargs=open_kwargs) as actual:
            for k, v in actual.variables.items():
                assert v._in_memory == (k in actual.dims)
                assert v.chunks == rechunked[k].chunks
    chunks = {'dim1': 2, 'dim2': 3, 'dim3': 5}
    rechunked = original.chunk(chunks=chunks)
    open_kwargs = {'chunks': chunks, 'backend_kwargs': {'overwrite_encoded_chunks': True}}
    with self.roundtrip(original, open_kwargs=open_kwargs) as actual:
        for k, v in actual.variables.items():
            assert v.chunks == rechunked[k].chunks
        with self.roundtrip(actual) as auto:
            for k, v in actual.variables.items():
                assert v.chunks == rechunked[k].chunks
            assert_identical(actual, auto)
            assert_identical(actual.load(), auto.load())