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
@pytest.mark.parametrize('index_dtype', ['int32', 'uint32'])
def test_gather_add(ops, dtype, index_dtype):
    table = ops.xp.arange(12, dtype=dtype).reshape(4, 3)
    indices = ops.xp.array([[0, 2], [3, 1], [0, 1]], dtype=index_dtype)
    gathered = ops.gather_add(table, indices)
    ops.xp.testing.assert_allclose(gathered, [[6.0, 8.0, 10.0], [12.0, 14.0, 16.0], [3.0, 5.0, 7.0]])