from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import copy, deepcopy
from io import StringIO
from textwrap import dedent
from typing import Any, Literal
import numpy as np
import pandas as pd
import pytest
from pandas.core.indexes.datetimes import DatetimeIndex
import xarray as xr
from xarray import (
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core import dtypes, indexing, utils
from xarray.core.common import duck_array_ops, full_like
from xarray.core.coordinates import Coordinates, DatasetCoordinates
from xarray.core.indexes import Index, PandasIndex
from xarray.core.utils import is_scalar
from xarray.namedarray.pycompat import array_type, integer_types
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
@requires_sparse
def test_from_dataframe_sparse(self) -> None:
    import sparse
    df_base = pd.DataFrame({'x': range(10), 'y': list('abcdefghij'), 'z': np.arange(0, 100, 10)})
    ds_sparse = Dataset.from_dataframe(df_base.set_index('x'), sparse=True)
    ds_dense = Dataset.from_dataframe(df_base.set_index('x'), sparse=False)
    assert isinstance(ds_sparse['y'].data, sparse.COO)
    assert isinstance(ds_sparse['z'].data, sparse.COO)
    ds_sparse['y'].data = ds_sparse['y'].data.todense()
    ds_sparse['z'].data = ds_sparse['z'].data.todense()
    assert_identical(ds_dense, ds_sparse)
    ds_sparse = Dataset.from_dataframe(df_base.set_index(['x', 'y']), sparse=True)
    ds_dense = Dataset.from_dataframe(df_base.set_index(['x', 'y']), sparse=False)
    assert isinstance(ds_sparse['z'].data, sparse.COO)
    ds_sparse['z'].data = ds_sparse['z'].data.todense()
    assert_identical(ds_dense, ds_sparse)