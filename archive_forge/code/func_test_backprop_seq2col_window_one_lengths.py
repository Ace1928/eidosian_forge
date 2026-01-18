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
def test_backprop_seq2col_window_one_lengths(ops, dtype):
    d_y = ops.xp.arange(0.1, 4.6, step=0.1, dtype=dtype).reshape(5, 9)
    lengths = ops.asarray1i([1, 3, 1])
    d_seqs = ops.backprop_seq2col(d_y, 1, lengths=lengths)
    ops.xp.testing.assert_allclose(ops.asarray2f([[0.4, 0.5, 0.6], [3.2, 3.4, 3.6], [6.6, 6.9, 7.2], [5.6, 5.8, 6.0], [4.0, 4.1, 4.2]], dtype=dtype), d_seqs, atol=1e-06)