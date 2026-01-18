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
def test_from_dataframe_categorical(self) -> None:
    cat = pd.CategoricalDtype(categories=['foo', 'bar', 'baz', 'qux', 'quux', 'corge'])
    i1 = pd.Series(['foo', 'bar', 'foo'], dtype=cat)
    i2 = pd.Series(['bar', 'bar', 'baz'], dtype=cat)
    df = pd.DataFrame({'i1': i1, 'i2': i2, 'values': [1, 2, 3]})
    ds = df.set_index('i1').to_xarray()
    assert len(ds['i1']) == 3
    ds = df.set_index(['i1', 'i2']).to_xarray()
    assert len(ds['i1']) == 2
    assert len(ds['i2']) == 2