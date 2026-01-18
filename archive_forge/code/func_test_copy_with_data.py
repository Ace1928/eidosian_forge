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
def test_copy_with_data(self) -> None:
    orig = Variable(('x', 'y'), [[1.5, 2.0], [3.1, 4.3]], {'foo': 'bar'})
    new_data = np.array([[2.5, 5.0], [7.1, 43]])
    actual = orig.copy(data=new_data)
    expected = orig.copy()
    expected.data = new_data
    assert_identical(expected, actual)