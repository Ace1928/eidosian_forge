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
def test_inplace_math(self):
    x = np.arange(5)
    v = Variable(['x'], x)
    v2 = v
    v2 += 1
    assert v is v2
    assert source_ndarray(v.values) is x
    assert_array_equal(v.values, np.arange(5) + 1)
    with pytest.raises(ValueError, match='dimensions cannot change'):
        v += Variable('y', np.arange(5))