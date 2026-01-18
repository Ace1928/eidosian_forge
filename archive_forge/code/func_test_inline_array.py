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
@pytest.mark.skipif(ON_WINDOWS, reason='counting number of tasks in graph fails on windows for some reason')
def test_inline_array(self) -> None:
    with create_tmp_file() as tmp:
        original = Dataset({'foo': ('x', np.random.randn(10))})
        original.to_netcdf(tmp)
        chunks = {'time': 10}

        def num_graph_nodes(obj):
            return len(obj.__dask_graph__())
        with open_dataset(tmp, inline_array=False, chunks=chunks) as not_inlined_ds, open_dataset(tmp, inline_array=True, chunks=chunks) as inlined_ds:
            assert num_graph_nodes(inlined_ds) < num_graph_nodes(not_inlined_ds)
        with open_dataarray(tmp, inline_array=False, chunks=chunks) as not_inlined_da, open_dataarray(tmp, inline_array=True, chunks=chunks) as inlined_da:
            assert num_graph_nodes(inlined_da) < num_graph_nodes(not_inlined_da)