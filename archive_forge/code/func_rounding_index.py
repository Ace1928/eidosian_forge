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
def rounding_index(date_type):
    return xr.CFTimeIndex([date_type(1, 1, 1, 1, 59, 59, 999512), date_type(1, 1, 1, 3, 0, 1, 500001), date_type(1, 1, 1, 7, 0, 6, 499999)])