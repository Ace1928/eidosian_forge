from __future__ import annotations
import functools
import operator
import pickle
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import xarray as xr
from xarray.core.alignment import broadcast
from xarray.core.computation import (
from xarray.tests import (
def np_corr_ind(ts1, ts2, a, x):
    ts1, ts2 = broadcast(ts1, ts2)
    valid_values = ts1.notnull() & ts2.notnull()
    ts1 = ts1.where(valid_values)
    ts2 = ts2.where(valid_values)
    return np.ma.corrcoef(np.ma.masked_invalid(ts1.sel(a=a, x=x).data.flatten()), np.ma.masked_invalid(ts2.sel(a=a, x=x).data.flatten()))[0, 1]