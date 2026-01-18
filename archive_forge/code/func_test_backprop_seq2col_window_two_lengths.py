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
def test_backprop_seq2col_window_two_lengths(ops, dtype):
    d_y = ops.xp.arange(0.1, 7.6, step=0.1, dtype=dtype).reshape(5, 15)
    lengths = ops.asarray1i([1, 3, 1])
    d_seqs = ops.backprop_seq2col(d_y, 2, lengths=lengths)
    ops.xp.testing.assert_allclose(ops.asarray2f([[0.7, 0.8, 0.9], [10.2, 10.5, 10.8], [11.1, 11.4, 11.7], [12.0, 12.3, 12.6], [6.7, 6.8, 6.9]], dtype=dtype), d_seqs)