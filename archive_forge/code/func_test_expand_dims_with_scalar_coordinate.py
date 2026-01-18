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
def test_expand_dims_with_scalar_coordinate(self) -> None:
    array = DataArray(np.random.randn(3, 4), dims=['x', 'dim_0'], coords={'x': np.linspace(0.0, 1.0, 3), 'z': 1.0}, attrs={'key': 'entry'})
    actual = array.expand_dims(dim='z')
    expected = DataArray(np.expand_dims(array.values, 0), dims=['z', 'x', 'dim_0'], coords={'x': np.linspace(0.0, 1.0, 3), 'z': np.ones(1)}, attrs={'key': 'entry'})
    assert_identical(expected, actual)
    roundtripped = actual.squeeze(['z'], drop=False)
    assert_identical(array, roundtripped)