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
def test_get_level_variable(self):
    midx = pd.MultiIndex.from_product([['a', 'b'], [1, 2]], names=['level_1', 'level_2'])
    x = IndexVariable('x', midx)
    level_1 = IndexVariable('x', midx.get_level_values('level_1'))
    assert_identical(x.get_level_variable('level_1'), level_1)
    with pytest.raises(ValueError, match='has no MultiIndex'):
        IndexVariable('y', [10.0]).get_level_variable('level')