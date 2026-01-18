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
@pytest.fixture(params=_CFTIME_CALENDARS)
def date_type(request):
    return _all_cftime_date_types()[request.param]