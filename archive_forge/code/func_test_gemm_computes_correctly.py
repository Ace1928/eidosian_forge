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
@pytest.mark.parametrize('cpu_ops', [*CPU_OPS, BLIS_OPS])
def test_gemm_computes_correctly(cpu_ops):
    W = numpy.zeros((3, 2), dtype='f')
    X = numpy.zeros((4, 2), dtype='f')
    W += numpy.random.uniform(size=W.size).reshape(W.shape)
    X += numpy.random.uniform(size=X.size).reshape(X.shape)
    Y = cpu_ops.gemm(X, W, trans2=True)
    expected = numpy.dot(X, W.T)
    assert_allclose(expected, Y, atol=0.0001, rtol=0.0001)
    W = numpy.zeros((2, 3), dtype='f')
    X = numpy.zeros((2, 4), dtype='f')
    W += numpy.random.uniform(size=W.size).reshape(W.shape)
    X += numpy.random.uniform(size=X.size).reshape(X.shape)
    Y = cpu_ops.gemm(X, W, trans1=True)
    expected = numpy.dot(X.T, W)
    assert_allclose(expected, Y, atol=0.0001, rtol=0.0001)
    cpu_ops.gemm(X, W, trans1=True, out=Y)