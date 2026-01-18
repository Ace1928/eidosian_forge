from __future__ import annotations
import pickle
from datetime import timedelta
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import (
from xarray.tests import (
from xarray.tests.test_coding_times import (
@requires_cftime
@pytest.mark.parametrize(('name', 'expected_name'), [('bar', 'bar'), (None, 'foo')])
def test_constructor_with_name(index_with_name, name, expected_name):
    result = CFTimeIndex(index_with_name, name=name).name
    assert result == expected_name