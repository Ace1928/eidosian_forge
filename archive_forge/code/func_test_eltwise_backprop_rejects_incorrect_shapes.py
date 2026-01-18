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
@pytest.mark.parametrize('dtype', FLOAT_TYPES)
@pytest.mark.parametrize('op', ['backprop_clipped_linear', 'backprop_dish', 'backprop_gelu', 'backprop_gelu_approx', 'backprop_hard_sigmoid', 'backprop_hard_swish', 'backprop_hard_swish_mobilenet', 'backprop_hard_tanh', 'backprop_mish', 'backprop_relu', 'backprop_relu_k', 'backprop_softmax', 'backprop_swish'])
def test_eltwise_backprop_rejects_incorrect_shapes(ops, dtype, op):
    backprop = getattr(ops, op)
    positional_args = [p for p in inspect.signature(backprop).parameters.values() if p.default == inspect.Parameter.empty]
    if len(positional_args) == 3:
        with pytest.raises(ValueError):
            backprop(ops.xp.zeros(10, dtype=dtype), ops.xp.zeros(5, dtype=dtype), ops.xp.zeros(10, dtype=dtype))
        with pytest.raises(ValueError):
            backprop(ops.xp.zeros(10, dtype=dtype), ops.xp.zeros(10, dtype=dtype), ops.xp.zeros(5, dtype=dtype))
    else:
        with pytest.raises(ValueError):
            backprop(ops.xp.arange(-10, 10, dtype=dtype), ops.xp.arange(5, -5, -1, dtype=dtype))