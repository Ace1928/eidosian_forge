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
def test_hash_gives_distinct_keys(ops):
    ids = ops.alloc1f(5, dtype='uint64')
    keys = ops.hash(ids, 0)
    assert keys.shape == (5, 4)
    assert keys.dtype == 'uint32'
    for i in range(len(ids)):
        for j in range(keys.shape[1]):
            assert keys[i, j] != 0