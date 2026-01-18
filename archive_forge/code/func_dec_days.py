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
def dec_days(date_type):
    import cftime
    if date_type is cftime.Datetime360Day:
        return 30
    else:
        return 31