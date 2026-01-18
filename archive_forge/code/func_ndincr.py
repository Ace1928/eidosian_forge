import functools
import sys
import math
import warnings
import numpy as np
from .._utils import set_module
import numpy.core.numeric as _nx
from numpy.core.numeric import ScalarType, array
from numpy.core.numerictypes import issubdtype
import numpy.matrixlib as matrixlib
from .function_base import diff
from numpy.core.multiarray import ravel_multi_index, unravel_index
from numpy.core import overrides, linspace
from numpy.lib.stride_tricks import as_strided
def ndincr(self):
    """
        Increment the multi-dimensional index by one.

        This method is for backward compatibility only: do not use.

        .. deprecated:: 1.20.0
            This method has been advised against since numpy 1.8.0, but only
            started emitting DeprecationWarning as of this version.
        """
    warnings.warn('`ndindex.ndincr()` is deprecated, use `next(ndindex)` instead', DeprecationWarning, stacklevel=2)
    next(self)