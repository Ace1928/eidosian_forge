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
@pytest.mark.parametrize('ops', XP_OPS)
@pytest.mark.parametrize('depth,dirs,nO,batch_size,nI', [(1, 1, 1, 1, 1), (1, 1, 2, 1, 1), (1, 1, 2, 1, 2), (2, 1, 1, 1, 1), (2, 1, 2, 2, 2), (1, 2, 2, 1, 1), (2, 2, 2, 2, 2)])
def test_lstm_forward_training(ops, depth, dirs, nO, batch_size, nI):
    reference_ops = Ops()
    params, H0, C0, X, size_at_t = get_lstm_args(depth, dirs, nO, batch_size, nI)
    reference = reference_ops.lstm_forward_training(params, H0, C0, X, size_at_t)
    Y, fwd_state = ops.lstm_forward_training(params, H0, C0, X, size_at_t)
    assert_allclose(fwd_state[2], reference[1][2], atol=0.0001, rtol=0.001)
    assert_allclose(fwd_state[1], reference[1][1], atol=0.0001, rtol=0.001)
    assert_allclose(Y, reference[0], atol=0.0001, rtol=0.001)