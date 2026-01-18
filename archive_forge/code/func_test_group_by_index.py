from __future__ import annotations
import copy
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.indexes import (
from xarray.core.variable import IndexVariable, Variable
from xarray.tests import assert_array_equal, assert_identical, requires_cftime
from xarray.tests.test_coding_times import _all_cftime_date_types
def test_group_by_index(self, unique_indexes, indexes):
    expected = [(unique_indexes[0], {'x': indexes.variables['x']}), (unique_indexes[1], {'y': indexes.variables['y']}), (unique_indexes[2], {'z': indexes.variables['z'], 'one': indexes.variables['one'], 'two': indexes.variables['two']})]
    assert indexes.group_by_index() == expected