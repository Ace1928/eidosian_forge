from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import deepcopy
from textwrap import dedent
from typing import Any, Final, Literal, cast
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import (
from xarray.coding.times import CFDatetimeCoder
from xarray.core import dtypes
from xarray.core.common import full_like
from xarray.core.coordinates import Coordinates
from xarray.core.indexes import Index, PandasIndex, filter_indexes_from_coords
from xarray.core.types import QueryEngineOptions, QueryParserOptions
from xarray.core.utils import is_scalar
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
@requires_sparse
def test_from_multiindex_series_sparse(self) -> None:
    import sparse
    idx = pd.MultiIndex.from_product([np.arange(3), np.arange(5)], names=['a', 'b'])
    series = pd.Series(np.random.RandomState(0).random(len(idx)), index=idx).sample(n=5, random_state=3)
    dense = DataArray.from_series(series, sparse=False)
    expected_coords = sparse.COO.from_numpy(dense.data, np.nan).coords
    actual_sparse = xr.DataArray.from_series(series, sparse=True)
    actual_coords = actual_sparse.data.coords
    np.testing.assert_equal(actual_coords, expected_coords)