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
def date_dict(year=None, month=None, day=None, hour=None, minute=None, second=None):
    return dict(year=year, month=month, day=day, hour=hour, minute=minute, second=second)