from __future__ import annotations
import sys
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
from numpy.core import defchararray
import xarray as xr
from xarray.core import formatting
from xarray.tests import requires_cftime, requires_dask, requires_netCDF4
def test_diff_dataset_repr(self) -> None:
    ds_a = xr.Dataset(data_vars={'var1': (('x', 'y'), np.array([[1, 2, 3], [4, 5, 6]], dtype='int64')), 'var2': ('x', np.array([3, 4], dtype='int64'))}, coords={'x': ('x', np.array(['a', 'b'], dtype='U1'), {'foo': 'bar', 'same': 'same'}), 'y': np.array([1, 2, 3], dtype='int64')}, attrs={'title': 'mytitle', 'description': 'desc'})
    ds_b = xr.Dataset(data_vars={'var1': ('x', np.array([1, 2], dtype='int64'))}, coords={'x': ('x', np.array(['a', 'c'], dtype='U1'), {'source': 0, 'foo': 'baz', 'same': 'same'}), 'label': ('x', np.array([1, 2], dtype='int64'))}, attrs={'title': 'newtitle'})
    byteorder = '<' if sys.byteorder == 'little' else '>'
    expected = dedent("        Left and right Dataset objects are not identical\n        Differing dimensions:\n            (x: 2, y: 3) != (x: 2)\n        Differing coordinates:\n        L * x        (x) %cU1 8B 'a' 'b'\n            Differing variable attributes:\n                foo: bar\n        R * x        (x) %cU1 8B 'a' 'c'\n            Differing variable attributes:\n                source: 0\n                foo: baz\n        Coordinates only on the left object:\n          * y        (y) int64 24B 1 2 3\n        Coordinates only on the right object:\n            label    (x) int64 16B 1 2\n        Differing data variables:\n        L   var1     (x, y) int64 48B 1 2 3 4 5 6\n        R   var1     (x) int64 16B 1 2\n        Data variables only on the left object:\n            var2     (x) int64 16B 3 4\n        Differing attributes:\n        L   title: mytitle\n        R   title: newtitle\n        Attributes only on the left object:\n            description: desc" % (byteorder, byteorder))
    actual = formatting.diff_dataset_repr(ds_a, ds_b, 'identical')
    assert actual == expected