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
@pytest.mark.parametrize('order', ['C', 'F', 'A', 'K'])
def test_einsum_order(order):
    sig = 'ea,fb,abcd,gc,hd->efgh'
    input_sigs = sig.split('->')[0].split(',')
    np_inputs, da_inputs = _numpy_and_dask_inputs(input_sigs)
    assert_eq(np.einsum(sig, *np_inputs, order=order), da.einsum(sig, *np_inputs, order=order))