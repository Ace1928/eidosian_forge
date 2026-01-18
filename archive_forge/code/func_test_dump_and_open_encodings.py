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
def test_dump_and_open_encodings(self) -> None:
    with create_tmp_file() as tmp_file:
        with nc4.Dataset(tmp_file, 'w') as ds:
            ds.createDimension('time', size=10)
            ds.createVariable('time', np.int32, dimensions=('time',))
            units = 'days since 1999-01-01'
            ds.variables['time'].setncattr('units', units)
            ds.variables['time'][:] = np.arange(10) + 4
        with open_dataset(tmp_file) as xarray_dataset:
            with create_tmp_file() as tmp_file2:
                xarray_dataset.to_netcdf(tmp_file2)
                with nc4.Dataset(tmp_file2, 'r') as ds:
                    assert ds.variables['time'].getncattr('units') == units
                    assert_array_equal(ds.variables['time'], np.arange(10) + 4)