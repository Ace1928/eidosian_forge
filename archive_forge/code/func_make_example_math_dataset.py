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
def make_example_math_dataset(self):
    variables = {'bar': ('x', np.arange(100, 400, 100)), 'foo': (('x', 'y'), 1.0 * np.arange(12).reshape(3, 4))}
    coords = {'abc': ('x', ['a', 'b', 'c']), 'y': 10 * np.arange(4)}
    ds = Dataset(variables, coords)
    ds['foo'][0, 0] = np.nan
    return ds