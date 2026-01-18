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
def test_delete_coords() -> None:
    """Make sure that deleting a coordinate doesn't corrupt the DataArray.
    See issue #3899.

    Also test that deleting succeeds and produces the expected output.
    """
    a0 = DataArray(np.array([[1, 2, 3], [4, 5, 6]]), dims=['y', 'x'], coords={'x': ['a', 'b', 'c'], 'y': [-1, 1]})
    assert_identical(a0, a0)
    a1 = a0.copy()
    del a1.coords['y']
    assert_identical(a0, a0)
    assert a0.dims == ('y', 'x')
    assert a1.dims == ('y', 'x')
    assert set(a0.coords.keys()) == {'x', 'y'}
    assert set(a1.coords.keys()) == {'x'}