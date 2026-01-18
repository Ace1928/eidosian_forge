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
def test_more_transforms_pass_lazy_array_equiv(map_da, map_ds):
    with raise_if_dask_computes():
        assert_equal(map_ds.cxy.broadcast_like(map_ds.cxy), map_ds.cxy)
        assert_equal(xr.broadcast(map_ds.cxy, map_ds.cxy)[0], map_ds.cxy)
        assert_equal(map_ds.map(lambda x: x), map_ds)
        assert_equal(map_ds.set_coords('a').reset_coords('a'), map_ds)
        assert_equal(map_ds.assign({'a': map_ds.a}), map_ds)
        assert_equal(map_ds.rename_vars({'cxy': 'cnew'}).rename_vars({'cnew': 'cxy'}), map_ds)
        assert_equal(map_da._from_temp_dataset(map_da._to_temp_dataset()), map_da)
        assert_equal(map_da.astype(map_da.dtype), map_da)
        assert_equal(map_da.transpose('y', 'x', transpose_coords=False).cxy, map_da.cxy)