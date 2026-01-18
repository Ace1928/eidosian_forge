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
def test_coord_coords(self) -> None:
    orig = DataArray([10, 20], {'x': [1, 2], 'x2': ('x', ['a', 'b']), 'z': 4}, dims='x')
    actual = orig.coords['x']
    expected = DataArray([1, 2], {'z': 4, 'x2': ('x', ['a', 'b']), 'x': [1, 2]}, dims='x', name='x')
    assert_identical(expected, actual)
    del actual.coords['x2']
    assert_identical(expected.reset_coords('x2', drop=True), actual)
    actual.coords['x3'] = ('x', ['a', 'b'])
    expected = DataArray([1, 2], {'z': 4, 'x3': ('x', ['a', 'b']), 'x': [1, 2]}, dims='x', name='x')
    assert_identical(expected, actual)