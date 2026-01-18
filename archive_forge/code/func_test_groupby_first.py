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
def test_groupby_first(self):
    x = self.sp_xr.copy()
    x.coords['ab'] = ('x', ['a', 'a', 'b', 'b'])
    x.groupby('ab').first()
    x.groupby('ab').first(skipna=False)