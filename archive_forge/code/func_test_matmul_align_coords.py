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
def test_matmul_align_coords(self) -> None:
    x_a = np.arange(6)
    x_b = np.arange(2, 8)
    da_vals = np.arange(6)
    da_a = DataArray(da_vals, coords=[x_a], dims=['x'])
    da_b = DataArray(da_vals, coords=[x_b], dims=['x'])
    result = da_a @ da_b
    expected = da_a.dot(da_b)
    assert_identical(result, expected)
    with xr.set_options(arithmetic_join='exact'):
        with pytest.raises(ValueError, match='cannot align.*join.*exact.*not equal.*'):
            da_a @ da_b