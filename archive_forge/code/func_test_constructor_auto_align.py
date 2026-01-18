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
def test_constructor_auto_align(self) -> None:
    a = DataArray([1, 2], [('x', [0, 1])])
    b = DataArray([3, 4], [('x', [1, 2])])
    expected = Dataset({'a': ('x', [1, 2, np.nan]), 'b': ('x', [np.nan, 3, 4])}, {'x': [0, 1, 2]})
    actual = Dataset({'a': a, 'b': b})
    assert_identical(expected, actual)
    assert isinstance(actual.variables['x'], IndexVariable)
    c = ('y', [3, 4])
    expected2 = expected.merge({'c': c})
    actual = Dataset({'a': a, 'b': b, 'c': c})
    assert_identical(expected2, actual)
    d = ('x', [3, 2, 1])
    expected3 = expected.merge({'d': d})
    actual = Dataset({'a': a, 'b': b, 'd': d})
    assert_identical(expected3, actual)
    e = ('x', [0, 0])
    with pytest.raises(ValueError, match='conflicting sizes'):
        Dataset({'a': a, 'b': b, 'e': e})