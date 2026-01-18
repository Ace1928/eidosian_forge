from __future__ import annotations
import contextlib
import warnings
import numpy as np
import pandas as pd
import pytest
from xarray import (
from xarray.backends.common import WritableCFDataStore
from xarray.backends.memory import InMemoryDataStore
from xarray.conventions import decode_cf
from xarray.testing import assert_identical
from xarray.tests import (
from xarray.tests.test_backends import CFEncodedBase
def test_multidimensional_coordinates(self) -> None:
    zeros1 = np.zeros((1, 5, 3))
    zeros2 = np.zeros((1, 6, 3))
    zeros3 = np.zeros((1, 5, 4))
    orig = Dataset({'lon1': (['x1', 'y1'], zeros1.squeeze(0), {}), 'lon2': (['x2', 'y1'], zeros2.squeeze(0), {}), 'lon3': (['x1', 'y2'], zeros3.squeeze(0), {}), 'lat1': (['x1', 'y1'], zeros1.squeeze(0), {}), 'lat2': (['x2', 'y1'], zeros2.squeeze(0), {}), 'lat3': (['x1', 'y2'], zeros3.squeeze(0), {}), 'foo1': (['time', 'x1', 'y1'], zeros1, {'coordinates': 'lon1 lat1'}), 'foo2': (['time', 'x2', 'y1'], zeros2, {'coordinates': 'lon2 lat2'}), 'foo3': (['time', 'x1', 'y2'], zeros3, {'coordinates': 'lon3 lat3'}), 'time': ('time', [0.0], {'units': 'hours since 2017-01-01'})})
    orig = conventions.decode_cf(orig)
    enc, attrs = conventions.encode_dataset_coordinates(orig)
    foo1_coords = enc['foo1'].attrs.get('coordinates', '')
    foo2_coords = enc['foo2'].attrs.get('coordinates', '')
    foo3_coords = enc['foo3'].attrs.get('coordinates', '')
    assert foo1_coords == 'lon1 lat1'
    assert foo2_coords == 'lon2 lat2'
    assert foo3_coords == 'lon3 lat3'
    assert 'coordinates' not in attrs