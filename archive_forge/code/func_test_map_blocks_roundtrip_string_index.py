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
def test_map_blocks_roundtrip_string_index():
    ds = xr.Dataset({'data': (['label'], [1, 2, 3])}, coords={'label': ['foo', 'bar', 'baz']}).chunk(label=1)
    assert ds.label.dtype == np.dtype('<U3')
    mapped = ds.map_blocks(lambda x: x, template=ds)
    assert mapped.label.dtype == ds.label.dtype
    mapped = ds.map_blocks(lambda x: x, template=None)
    assert mapped.label.dtype == ds.label.dtype
    mapped = ds.data.map_blocks(lambda x: x, template=ds.data)
    assert mapped.label.dtype == ds.label.dtype
    mapped = ds.data.map_blocks(lambda x: x, template=None)
    assert mapped.label.dtype == ds.label.dtype