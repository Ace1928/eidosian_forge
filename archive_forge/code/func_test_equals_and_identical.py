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
def test_equals_and_identical(self):
    d = np.random.rand(10, 3)
    d[0, 0] = np.nan
    v1 = Variable(('dim1', 'dim2'), data=d, attrs={'att1': 3, 'att2': [1, 2, 3]})
    v2 = Variable(('dim1', 'dim2'), data=d, attrs={'att1': 3, 'att2': [1, 2, 3]})
    assert v1.equals(v2)
    assert v1.identical(v2)
    v3 = Variable(('dim1', 'dim3'), data=d)
    assert not v1.equals(v3)
    v4 = Variable(('dim1', 'dim2'), data=d)
    assert v1.equals(v4)
    assert not v1.identical(v4)
    v5 = deepcopy(v1)
    v5.values[:] = np.random.rand(10, 3)
    assert not v1.equals(v5)
    assert not v1.equals(None)
    assert not v1.equals(d)
    assert not v1.identical(None)
    assert not v1.identical(d)