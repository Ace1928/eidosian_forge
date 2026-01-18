from __future__ import annotations
import itertools
import warnings
import pytest
from tlz import merge
import dask
import dask.array as da
from dask import config
from dask.array.chunk import getitem
from dask.array.slicing import (
from dask.array.utils import assert_eq, same_keys
@pytest.mark.parametrize('lock', [True, False])
@pytest.mark.parametrize('asarray', [True, False])
@pytest.mark.parametrize('fancy', [True, False])
def test_gh4043(lock, asarray, fancy):
    a1 = da.from_array(np.zeros(3), chunks=1, asarray=asarray, lock=lock, fancy=fancy)
    a2 = da.from_array(np.ones(3), chunks=1, asarray=asarray, lock=lock, fancy=fancy)
    al = da.stack([a1, a2])
    assert_eq(al, al)