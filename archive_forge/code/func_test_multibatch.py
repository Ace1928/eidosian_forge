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
def test_multibatch():
    fix_random_seed(0)
    ops = get_current_ops()
    arr1 = numpy.asarray([1, 2, 3, 4])
    arr2 = numpy.asarray([5, 6, 7, 8])
    batches = list(ops.multibatch(2, arr1, arr2))
    assert numpy.concatenate(batches).tolist() == [[1, 2], [5, 6], [3, 4], [7, 8]]
    batches = list(ops.multibatch(2, arr1, arr2, shuffle=True))
    assert len(batches) == 2
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    batches = list(ops.multibatch(2, [1, 2, 3, 4], [5, 6, 7, 8]))
    assert batches == [[[1, 2], [5, 6]], [[3, 4], [7, 8]]]
    with pytest.raises(ValueError):
        ops.multibatch(10, (i for i in range(100)), (i for i in range(100)))
    with pytest.raises(ValueError):
        ops.multibatch(10, arr1, (i for i in range(100)), arr2)