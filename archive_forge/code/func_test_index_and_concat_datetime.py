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
def test_index_and_concat_datetime(self):
    date_range = pd.date_range('2011-09-01', periods=10)
    for dates in [date_range, date_range.values, date_range.to_pydatetime()]:
        expected = self.cls('t', dates)
        for times in [[expected[i] for i in range(10)], [expected[i:i + 1] for i in range(10)], [expected[[i]] for i in range(10)]]:
            actual = Variable.concat(times, 't')
            assert expected.dtype == actual.dtype
            assert_array_equal(expected, actual)