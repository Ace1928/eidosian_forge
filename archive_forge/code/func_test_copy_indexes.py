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
def test_copy_indexes(self, indexes) -> None:
    copied, index_vars = indexes.copy_indexes()
    assert copied.keys() == indexes.keys()
    for new, original in zip(copied.values(), indexes.values()):
        assert new.equals(original)
    assert copied['z'] is copied['one'] is copied['two']
    assert index_vars.keys() == indexes.variables.keys()
    for new, original in zip(index_vars.values(), indexes.variables.values()):
        assert_identical(new, original)