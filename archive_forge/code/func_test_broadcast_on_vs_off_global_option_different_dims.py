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
def test_broadcast_on_vs_off_global_option_different_dims(self) -> None:
    xda_1 = xr.DataArray([1], dims='x1')
    xda_2 = xr.DataArray([1], dims='x2')
    with xr.set_options(arithmetic_broadcast=True):
        expected_xda = xr.DataArray([[1.0]], dims=('x1', 'x2'))
        actual_xda = xda_1 / xda_2
        assert_identical(actual_xda, expected_xda)
    with xr.set_options(arithmetic_broadcast=False):
        with pytest.raises(ValueError, match=re.escape("Broadcasting is necessary but automatic broadcasting is disabled via global option `'arithmetic_broadcast'`. Use `xr.set_options(arithmetic_broadcast=True)` to enable automatic broadcasting.")):
            xda_1 / xda_2