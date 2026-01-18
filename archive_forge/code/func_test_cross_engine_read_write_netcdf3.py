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
def test_cross_engine_read_write_netcdf3(self) -> None:
    data = create_test_data()
    valid_engines: set[T_NetcdfEngine] = set()
    if has_netCDF4:
        valid_engines.add('netcdf4')
    if has_scipy:
        valid_engines.add('scipy')
    for write_engine in valid_engines:
        for format in self.netcdf3_formats:
            with create_tmp_file() as tmp_file:
                data.to_netcdf(tmp_file, format=format, engine=write_engine)
                for read_engine in valid_engines:
                    with open_dataset(tmp_file, engine=read_engine) as actual:
                        [assert_allclose(data[k].variable, actual[k].variable) for k in data.variables]