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
@requires_netCDF4
def test_encoding_enum__no_fill_value(self):
    with create_tmp_file() as tmp_file:
        cloud_type_dict = {'clear': 0, 'cloudy': 1}
        with nc4.Dataset(tmp_file, mode='w') as nc:
            nc.createDimension('time', size=2)
            cloud_type = nc.createEnumType('u1', 'cloud_type', cloud_type_dict)
            v = nc.createVariable('clouds', cloud_type, 'time', fill_value=None)
            v[:] = 1
        with open_dataset(tmp_file) as original:
            save_kwargs = {}
            if self.engine == 'h5netcdf':
                save_kwargs['invalid_netcdf'] = True
            with self.roundtrip(original, save_kwargs=save_kwargs) as actual:
                assert_equal(original, actual)
                assert actual.clouds.encoding['dtype'].metadata['enum'] == cloud_type_dict
                if self.engine != 'h5netcdf':
                    assert actual.clouds.encoding['dtype'].metadata['enum_name'] == 'cloud_type'