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
@property
def cached_code(self):
    """Returns `next(iter(self.cached_codes.values()))`.

        This proprety method is for debugging purpose.
        The return value is not guaranteed to keep backward compatibility.
        """
    codes = self.cached_codes
    if len(codes) > 1:
        warnings.warn('The input types of the kernel could not be inferred. Please use `.cached_codes` instead.')
    return next(iter(codes.values()))