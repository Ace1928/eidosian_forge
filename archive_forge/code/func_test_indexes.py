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
def test_indexes(self) -> None:
    array = DataArray(np.zeros((2, 3)), [('x', [0, 1]), ('y', ['a', 'b', 'c'])])
    expected_indexes = {'x': pd.Index([0, 1]), 'y': pd.Index(['a', 'b', 'c'])}
    expected_xindexes = {k: PandasIndex(idx, k) for k, idx in expected_indexes.items()}
    assert array.xindexes.keys() == expected_xindexes.keys()
    assert array.indexes.keys() == expected_indexes.keys()
    assert all([isinstance(idx, pd.Index) for idx in array.indexes.values()])
    assert all([isinstance(idx, Index) for idx in array.xindexes.values()])
    for k in expected_indexes:
        assert array.xindexes[k].equals(expected_xindexes[k])
        assert array.indexes[k].equals(expected_indexes[k])