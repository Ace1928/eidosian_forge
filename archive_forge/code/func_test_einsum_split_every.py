from __future__ import annotations
import contextlib
import itertools
import pickle
import sys
import warnings
from numbers import Number
import pytest
import dask
from dask.delayed import delayed
import dask.array as da
from dask.array.numpy_compat import NUMPY_GE_123, NUMPY_GE_200, AxisError
from dask.array.utils import assert_eq, same_keys
@pytest.mark.parametrize('split_every', [None, 2])
def test_einsum_split_every(split_every):
    np_inputs, da_inputs = _numpy_and_dask_inputs('a')
    assert_eq(np.einsum('a', *np_inputs), da.einsum('a', *da_inputs, split_every=split_every))