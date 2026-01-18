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
def test_expand_dims_int(self) -> None:
    original = Dataset({'x': ('a', np.random.randn(3)), 'y': (['b', 'a'], np.random.randn(4, 3))}, coords={'a': np.linspace(0, 1, 3), 'b': np.linspace(0, 1, 4), 'c': np.linspace(0, 1, 5)}, attrs={'key': 'entry'})
    actual = original.expand_dims(['z'], [1])
    expected = Dataset({'x': original['x'].expand_dims('z', 1), 'y': original['y'].expand_dims('z', 1)}, coords={'a': np.linspace(0, 1, 3), 'b': np.linspace(0, 1, 4), 'c': np.linspace(0, 1, 5)}, attrs={'key': 'entry'})
    assert_identical(expected, actual)
    roundtripped = actual.squeeze('z')
    assert_identical(original, roundtripped)
    actual = original.expand_dims(['z'], [-1])
    expected = Dataset({'x': original['x'].expand_dims('z', -1), 'y': original['y'].expand_dims('z', -1)}, coords={'a': np.linspace(0, 1, 3), 'b': np.linspace(0, 1, 4), 'c': np.linspace(0, 1, 5)}, attrs={'key': 'entry'})
    assert_identical(expected, actual)
    roundtripped = actual.squeeze('z')
    assert_identical(original, roundtripped)