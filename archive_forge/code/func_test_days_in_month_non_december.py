from __future__ import annotations
from itertools import product
from typing import Callable, Literal
import numpy as np
import pandas as pd
import pytest
from xarray import CFTimeIndex
from xarray.coding.cftime_offsets import (
from xarray.coding.frequencies import infer_freq
from xarray.core.dataarray import DataArray
from xarray.tests import (
def test_days_in_month_non_december(calendar):
    date_type = get_date_type(calendar)
    reference = date_type(1, 4, 1)
    assert _days_in_month(reference) == 30