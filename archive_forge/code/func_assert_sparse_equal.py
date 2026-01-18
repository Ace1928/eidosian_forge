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
def assert_sparse_equal(a, b):
    assert isinstance(a, sparse_array_type)
    assert isinstance(b, sparse_array_type)
    np.testing.assert_equal(a.todense(), b.todense())