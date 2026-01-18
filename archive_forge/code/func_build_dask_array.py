from __future__ import annotations
import operator
import pickle
import sys
from contextlib import suppress
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core import duck_array_ops
from xarray.core.duck_array_ops import lazy_array_equiv
from xarray.testing import assert_chunks_equal
from xarray.tests import (
from xarray.tests.test_backends import create_tmp_file
def build_dask_array(name):
    global kernel_call_count
    kernel_call_count = 0
    return dask.array.Array(dask={(name, 0): (kernel, name)}, name=name, chunks=((1,),), dtype=np.int64)