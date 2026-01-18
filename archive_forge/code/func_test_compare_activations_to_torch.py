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
@pytest.mark.skipif(not has_torch, reason='needs PyTorch')
@pytest.mark.skipif(torch_version < Version('1.9.0'), reason='needs PyTorch 1.9.0')
@pytest.mark.parametrize('ops', ALL_OPS)
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('torch_func', TORCH_FUNCS)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(x=strategies.floats(min_value=-30, max_value=30), dY=strategies.floats(min_value=-1, max_value=1))
def test_compare_activations_to_torch(ops, dtype, x, dY, torch_func):
    import torch
    func_name, pytorch_func = torch_func
    forward = getattr(ops, func_name)
    backward = getattr(ops, 'backprop_' + func_name)
    x_thinc = ops.asarray([x], dtype=dtype)
    x_torch = xp2torch(x_thinc, requires_grad=True)
    y = pytorch_func(x_torch)
    y_thinc = forward(x_thinc)
    y.backward()
    assert x_thinc.dtype == y_thinc.dtype
    assert y_thinc is not x_thinc
    y_think_inplace = forward(x_thinc, inplace=True)
    assert y_think_inplace is x_thinc
    assert ops.xp.isclose(y_thinc, y_think_inplace, atol=1e-06)
    assert ops.xp.isclose(y_thinc, y.detach(), atol=1e-05)
    x_thinc = ops.asarray([x], dtype=dtype)
    dY_thinc = ops.asarray([dY], dtype=dtype)
    dY_thinc_inplace = dY_thinc.copy()
    s = inspect.signature(backward)
    params = {p for p in s.parameters if p in ['dY', 'X', 'Y']}
    if params == {'dY', 'X', 'Y'}:
        dx_thinc = backward(dY_thinc, Y=y_thinc, X=x_thinc)
        assert dx_thinc.dtype == x_thinc.dtype
        assert dx_thinc is not dY_thinc
        dx_thinc_inplace = backward(dY=dY_thinc_inplace, Y=y_thinc, X=x_thinc, inplace=True)
        assert dx_thinc_inplace is dY_thinc_inplace
        assert ops.xp.isclose(dx_thinc, dx_thinc_inplace)
        assert ops.xp.isclose(x_torch.grad.item() * dY, float(dx_thinc), atol=1e-06)
    elif params == {'Y', 'dY'}:
        dx_thinc = backward(dY_thinc, Y=y_thinc)
        assert dx_thinc.dtype == x_thinc.dtype
        assert ops.xp.isclose(dx_thinc, backward(dY=dY_thinc_inplace, Y=y_thinc, inplace=True))
        assert ops.xp.isclose(x_torch.grad.item() * dY, float(dx_thinc), atol=1e-06)
    elif params == {'dY', 'X'}:
        dx_thinc = backward(dY_thinc, X=x_thinc)
        assert dx_thinc.dtype == x_thinc.dtype
        assert ops.xp.isclose(dx_thinc, backward(dY=dY_thinc_inplace, X=x_thinc, inplace=True))
        assert ops.xp.isclose(x_torch.grad.item() * dY, float(backward(dY_thinc, X=x_thinc)), atol=1e-06)
    else:
        raise NotImplementedError(f'No PyTorch comparison implemented for parameter set: {params}')