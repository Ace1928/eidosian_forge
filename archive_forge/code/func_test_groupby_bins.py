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
@pytest.mark.xfail(reason='Groupby reductions produce dense output')
def test_groupby_bins(self):
    x1 = self.ds_xr
    x2 = self.sp_xr
    m1 = x1.groupby_bins('x', bins=[0, 3, 7, 10]).sum(...)
    m2 = x2.groupby_bins('x', bins=[0, 3, 7, 10]).sum(...)
    assert isinstance(m2.data, sparse.SparseArray)
    assert np.allclose(m1.data, m2.data.todense())