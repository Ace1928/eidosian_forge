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
def test_constructor_pandas_sequence(self) -> None:
    ds = self.make_example_math_dataset()
    pandas_objs = {var_name: ds[var_name].to_pandas() for var_name in ['foo', 'bar']}
    ds_based_on_pandas = Dataset(pandas_objs, ds.coords, attrs=ds.attrs)
    del ds_based_on_pandas['x']
    assert_equal(ds, ds_based_on_pandas)
    rearranged_index = reversed(pandas_objs['foo'].index)
    pandas_objs['foo'] = pandas_objs['foo'].reindex(rearranged_index)
    ds_based_on_pandas = Dataset(pandas_objs, ds.coords, attrs=ds.attrs)
    del ds_based_on_pandas['x']
    assert_equal(ds, ds_based_on_pandas)