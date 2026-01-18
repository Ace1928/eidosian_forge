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
def index_with_name(date_type):
    dates = [date_type(1, 1, 1), date_type(1, 2, 1), date_type(2, 1, 1), date_type(2, 2, 1)]
    return CFTimeIndex(dates, name='foo')