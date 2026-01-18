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
@pytest.mark.parametrize('obj', [make_da(), make_ds()])
def test_map_blocks_convert_args_to_list(obj):
    expected = obj + 10
    with raise_if_dask_computes():
        actual = xr.map_blocks(operator.add, obj, [10])
    assert_chunks_equal(expected.chunk(), actual)
    assert_identical(actual, expected)