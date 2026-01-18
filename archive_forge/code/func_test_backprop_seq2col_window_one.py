import inspect
import platform
from typing import Tuple, cast
import numpy
import pytest
from hypothesis import given, settings
from hypothesis.strategies import composite, integers
from numpy.testing import assert_allclose
from packaging.version import Version
from thinc.api import (
from thinc.backends._custom_kernels import KERNELS, KERNELS_LIST, compile_mmh
from thinc.compat import has_cupy_gpu, has_torch, torch_version
from thinc.types import Floats2d
from thinc.util import torch2xp, xp2torch
from .. import strategies
from ..strategies import arrays_BI, ndarrays_of_shape
@pytest.mark.parametrize('ops', ALL_OPS)
@pytest.mark.parametrize('dtype', FLOAT_TYPES)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(X=strategies.arrays_BI())
def test_backprop_seq2col_window_one(ops, dtype, X):
    if X.shape[1] % 3:
        return None
    X = ops.asarray(X, dtype=dtype)
    if ops.xp.abs(X).max() >= 30:
        return None
    base_ops = Ops()
    base_ops.xp = ops.xp
    target = base_ops.backprop_seq2col(X, nW=1)
    predicted = ops.backprop_seq2col(X, nW=1)
    for row in range(target.shape[0]):
        diff = target[row].sum() - predicted[row].sum()
        if diff < -0.1 or diff > 0.1:
            print(row, diff)
            print(target[row])
            print(predicted[row])
    ops.xp.testing.assert_allclose(target, predicted, atol=0.001, rtol=0.001)