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
def test_minimize_graph_size():
    ds = Dataset({'foo': (('x', 'y', 'z'), dask.array.ones((120, 120, 120), chunks=(20, 20, 1)))}, coords={'x': np.arange(120), 'y': np.arange(120), 'z': np.arange(120)})
    mapped = ds.map_blocks(lambda x: x)
    graph = dict(mapped.__dask_graph__())
    numchunks = {k: len(v) for k, v in ds.chunksizes.items()}
    for var in 'xyz':
        actual = len([key for key in graph if var in key[0]])
        assert actual == numchunks[var], (actual, numchunks[var])