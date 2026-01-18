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
def test_avoid_excess_metadata_calls(self) -> None:
    """Test that chunk requests do not trigger redundant metadata requests.

        This test targets logic in backends.zarr.ZarrArrayWrapper, asserting that calls
        to retrieve chunk data after initialization do not trigger additional
        metadata requests.

        https://github.com/pydata/xarray/issues/8290
        """
    import zarr
    ds = xr.Dataset(data_vars={'test': (('Z',), np.array([123]).reshape(1))})
    Group = zarr.hierarchy.Group
    with self.create_zarr_target() as store, patch.object(Group, '__getitem__', side_effect=Group.__getitem__, autospec=True) as mock:
        ds.to_zarr(store, mode='w')
        xrds = xr.open_zarr(store)
        call_count = mock.call_count
        assert call_count == 1
        xrds.test.compute(scheduler='sync')
        assert mock.call_count == call_count