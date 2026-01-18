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
def test_map_blocks_template_convert_object():
    da = make_da()
    func = lambda x: x.to_dataset().isel(x=[1])
    template = da.to_dataset().isel(x=[1, 5, 9])
    with raise_if_dask_computes():
        actual = xr.map_blocks(func, da, template=template)
    assert_identical(actual, template)
    ds = da.to_dataset()
    func = lambda x: x.to_dataarray().isel(x=[1])
    template = ds.to_dataarray().isel(x=[1, 5, 9])
    with raise_if_dask_computes():
        actual = xr.map_blocks(func, ds, template=template)
    assert_identical(actual, template)