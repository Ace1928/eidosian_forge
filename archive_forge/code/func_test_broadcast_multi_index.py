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
def test_broadcast_multi_index(self) -> None:
    ds = Dataset({'foo': (('x', 'y', 'z'), np.ones((3, 4, 2)))}, {'x': ['a', 'b', 'c'], 'y': [1, 2, 3, 4]})
    stacked = ds.stack(space=['x', 'y'])
    broadcasted, _ = broadcast(stacked, stacked.space)
    assert broadcasted.xindexes['x'] is broadcasted.xindexes['space']
    assert broadcasted.xindexes['y'] is broadcasted.xindexes['space']