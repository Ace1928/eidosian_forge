import functools
import types
import warnings
import numpy as np
from .._utils import set_module
from . import multiarray as mu
from . import overrides
from . import umath as um
from . import numerictypes as nt
from .multiarray import asarray, array, asanyarray, concatenate
from . import _methods
@array_function_dispatch(_cumproduct_dispatcher, verify=False)
def cumproduct(*args, **kwargs):
    """
    Return the cumulative product over the given axis.

    .. deprecated:: 1.25.0
        ``cumproduct`` is deprecated as of NumPy 1.25.0, and will be
        removed in NumPy 2.0. Please use `cumprod` instead.

    See Also
    --------
    cumprod : equivalent function; see for details.
    """
    return cumprod(*args, **kwargs)