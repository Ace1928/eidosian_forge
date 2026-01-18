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
def test_encode_zarr_attr_value() -> None:
    arr = np.array([1, 2, 3])
    expected1 = [1, 2, 3]
    actual1 = backends.zarr.encode_zarr_attr_value(arr)
    assert isinstance(actual1, list)
    assert actual1 == expected1
    sarr = np.array(1)[()]
    expected2 = 1
    actual2 = backends.zarr.encode_zarr_attr_value(sarr)
    assert isinstance(actual2, int)
    assert actual2 == expected2
    expected3 = 'foo'
    actual3 = backends.zarr.encode_zarr_attr_value(expected3)
    assert isinstance(actual3, str)
    assert actual3 == expected3