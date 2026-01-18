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
def test_encoding_mfdataset(self) -> None:
    original = Dataset({'foo': ('t', np.random.randn(10)), 't': ('t', pd.date_range(start='2010-01-01', periods=10, freq='1D'))})
    original.t.encoding['units'] = 'days since 2010-01-01'
    with create_tmp_file() as tmp1:
        with create_tmp_file() as tmp2:
            ds1 = original.isel(t=slice(5))
            ds2 = original.isel(t=slice(5, 10))
            ds1.t.encoding['units'] = 'days since 2010-01-01'
            ds2.t.encoding['units'] = 'days since 2000-01-01'
            ds1.to_netcdf(tmp1)
            ds2.to_netcdf(tmp2)
            with open_mfdataset([tmp1, tmp2], combine='nested') as actual:
                assert actual.t.encoding['units'] == original.t.encoding['units']
                assert actual.t.encoding['units'] == ds1.t.encoding['units']
                assert actual.t.encoding['units'] != ds2.t.encoding['units']