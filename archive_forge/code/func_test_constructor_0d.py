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
@pytest.mark.filterwarnings('ignore:Converting non-nanosecond')
def test_constructor_0d(self) -> None:
    expected = Dataset({'x': ([], 1)})
    for arg in [1, np.array(1), expected['x']]:
        actual = Dataset({'x': arg})
        assert_identical(expected, actual)

    class Arbitrary:
        pass
    d = pd.Timestamp('2000-01-01T12')
    args = [True, None, 3.4, np.nan, 'hello', b'raw', np.datetime64('2000-01-01'), d, d.to_pydatetime(), Arbitrary()]
    for arg in args:
        print(arg)
        expected = Dataset({'x': ([], arg)})
        actual = Dataset({'x': arg})
        assert_identical(expected, actual)