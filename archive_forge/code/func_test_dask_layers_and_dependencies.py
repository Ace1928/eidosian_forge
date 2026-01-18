from __future__ import annotations
import operator
import pickle
import sys
from contextlib import suppress
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core import duck_array_ops
from xarray.core.duck_array_ops import lazy_array_equiv
from xarray.testing import assert_chunks_equal
from xarray.tests import (
from xarray.tests.test_backends import create_tmp_file
def test_dask_layers_and_dependencies():
    ds = Dataset({'foo': ('x', range(5)), 'bar': ('x', range(5))}).chunk()
    x = dask.delayed(ds)
    assert set(x.__dask_graph__().dependencies).issuperset(ds.__dask_graph__().dependencies)
    assert set(x.foo.__dask_graph__().dependencies).issuperset(ds.__dask_graph__().dependencies)