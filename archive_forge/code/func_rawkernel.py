import functools
import warnings
import numpy
import cupy
from cupy._core import core
from cupyx.jit import _compile
from cupyx.jit import _cuda_typerules
from cupyx.jit import _cuda_types
from cupyx.jit import _internal_types
from cupyx.jit._cuda_types import Scalar
def rawkernel(*, mode='cuda', device=False):
    """A decorator compiles a Python function into CUDA kernel.
    """
    cupy._util.experimental('cupyx.jit.rawkernel')

    def wrapper(func):
        return functools.update_wrapper(_JitRawKernel(func, mode, device), func)
    return wrapper