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
def test_map_blocks_errors_bad_template(obj):
    with pytest.raises(ValueError, match='unexpected coordinate variables'):
        xr.map_blocks(lambda x: x.assign_coords(a=10), obj, template=obj).compute()
    with pytest.raises(ValueError, match='does not contain coordinate variables'):
        xr.map_blocks(lambda x: x.drop_vars('cxy'), obj, template=obj).compute()
    with pytest.raises(ValueError, match="Dimensions {'x'} missing"):
        xr.map_blocks(lambda x: x.isel(x=1), obj, template=obj).compute()
    with pytest.raises(ValueError, match="Received dimension 'x' of length 1"):
        xr.map_blocks(lambda x: x.isel(x=[1]), obj, template=obj).compute()
    with pytest.raises(TypeError, match='must be a DataArray'):
        xr.map_blocks(lambda x: x.isel(x=[1]), obj, template=(obj,)).compute()
    with pytest.raises(ValueError, match='map_blocks requires that one block'):
        xr.map_blocks(lambda x: x.isel(x=[1]).assign_coords(x=10), obj, template=obj.isel(x=[1])).compute()
    with pytest.raises(ValueError, match="Expected index 'x' to be"):
        xr.map_blocks(lambda a: a.isel(x=[1]).assign_coords(x=[120]), obj, template=obj.isel(x=[1, 5, 9])).compute()