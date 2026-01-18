from __future__ import annotations
import warnings
from abc import ABC
from copy import copy, deepcopy
from datetime import datetime, timedelta
from textwrap import dedent
from typing import Generic
import numpy as np
import pandas as pd
import pytest
import pytz
from xarray import DataArray, Dataset, IndexVariable, Variable, set_options
from xarray.core import dtypes, duck_array_ops, indexing
from xarray.core.common import full_like, ones_like, zeros_like
from xarray.core.indexing import (
from xarray.core.types import T_DuckArray
from xarray.core.utils import NDArrayMixin
from xarray.core.variable import as_compatible_data, as_variable
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
from xarray.tests.test_namedarray import NamedArraySubclassobjects
@requires_dask
def test_full_like_dask(self) -> None:
    orig = Variable(dims=('x', 'y'), data=[[1.5, 2.0], [3.1, 4.3]], attrs={'foo': 'bar'}).chunk(dict(x=(1, 1), y=(2,)))

    def check(actual, expect_dtype, expect_values):
        assert actual.dtype == expect_dtype
        assert actual.shape == orig.shape
        assert actual.dims == orig.dims
        assert actual.attrs == orig.attrs
        assert actual.chunks == orig.chunks
        assert_array_equal(actual.values, expect_values)
    check(full_like(orig, 2), orig.dtype, np.full_like(orig.values, 2))
    check(full_like(orig, True, dtype=bool), bool, np.full_like(orig.values, True, dtype=bool))
    dsk = full_like(orig, 1).data.dask
    for v in dsk.values():
        if isinstance(v, tuple):
            for vi in v:
                assert not isinstance(vi, np.ndarray)
        else:
            assert not isinstance(v, np.ndarray)