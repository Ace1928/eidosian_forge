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
def test_merge_multiindex_level(self) -> None:
    data = create_test_multiindex()
    other = Dataset({'level_1': ('x', [0, 1])})
    with pytest.raises(ValueError, match='.*conflicting dimension sizes.*'):
        data.merge(other)
    other = Dataset({'level_1': ('x', range(4))})
    with pytest.raises(ValueError, match='unable to determine.*coordinates or not.*'):
        data.merge(other)
    other = Dataset(coords={'level_1': ('x', range(4))})
    assert_identical(data.merge(other), data)