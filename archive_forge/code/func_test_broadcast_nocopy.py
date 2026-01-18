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
def test_broadcast_nocopy(self) -> None:
    x = Dataset({'foo': (('x', 'y'), [[1, 1]])})
    y = Dataset({'bar': ('y', [2, 3])})
    actual_x, = broadcast(x)
    assert_identical(x, actual_x)
    assert source_ndarray(actual_x['foo'].data) is source_ndarray(x['foo'].data)
    actual_x, actual_y = broadcast(x, y)
    assert_identical(x, actual_x)
    assert source_ndarray(actual_x['foo'].data) is source_ndarray(x['foo'].data)