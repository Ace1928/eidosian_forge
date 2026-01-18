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
@pytest.fixture
def range_args(date_type):
    return ['0001', slice('0001-01-01', '0001-12-30'), slice(None, '0001-12-30'), slice(date_type(1, 1, 1), date_type(1, 12, 30)), slice(None, date_type(1, 12, 30))]