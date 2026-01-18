from __future__ import annotations
import math
import pickle
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Variable
from xarray.namedarray.pycompat import array_type
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_dataset_pickle(self):
    ds1 = xr.Dataset(data_vars={'a': ('x', sparse.COO.from_numpy(np.ones(4)))}, coords={'y': ('x', sparse.COO.from_numpy(np.arange(4)))})
    ds2 = pickle.loads(pickle.dumps(ds1))
    assert_identical(ds1, ds2)