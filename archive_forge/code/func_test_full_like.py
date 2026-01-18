from __future__ import annotations
import warnings
from abc import ABC
from copy import copy, deepcopy
from datetime import datetime, timedelta
from textwrap import dedent
from typing import Generic
import numpy as np
import pandas as pd
import pytest
import pytz
from xarray import DataArray, Dataset, IndexVariable, Variable, set_options
from xarray.core import dtypes, duck_array_ops, indexing
from xarray.core.common import full_like, ones_like, zeros_like
from xarray.core.indexing import (
from xarray.core.types import T_DuckArray
from xarray.core.utils import NDArrayMixin
from xarray.core.variable import as_compatible_data, as_variable
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
from xarray.tests.test_namedarray import NamedArraySubclassobjects
def test_full_like(self) -> None:
    orig = Variable(dims=('x', 'y'), data=[[1.5, 2.0], [3.1, 4.3]], attrs={'foo': 'bar'})
    expect = orig.copy(deep=True)
    expect.values = [[2.0, 2.0], [2.0, 2.0]]
    assert_identical(expect, full_like(orig, 2))
    expect.values = [[True, True], [True, True]]
    assert expect.dtype == bool
    assert_identical(expect, full_like(orig, True, dtype=bool))
    with pytest.raises(ValueError, match='must be scalar'):
        full_like(orig, [1.0, 2.0])
    with pytest.raises(ValueError, match="'dtype' cannot be dict-like"):
        full_like(orig, True, dtype={'x': bool})