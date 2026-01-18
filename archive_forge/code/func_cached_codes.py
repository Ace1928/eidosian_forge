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
def cached_codes(self):
    """Returns a dict that has input types as keys and codes values.

        This proprety method is for debugging purpose.
        The return value is not guaranteed to keep backward compatibility.
        """
    if len(self._cached_codes) == 0:
        warnings.warn('No codes are cached because compilation is deferred until the first function call.')
    return dict([(k, v.code) for k, v in self._cached_codes.items()])