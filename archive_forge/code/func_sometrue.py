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
@array_function_dispatch(_sometrue_dispatcher, verify=False)
def sometrue(*args, **kwargs):
    """
    Check whether some values are true.

    Refer to `any` for full documentation.

    .. deprecated:: 1.25.0
        ``sometrue`` is deprecated as of NumPy 1.25.0, and will be
        removed in NumPy 2.0. Please use `any` instead.

    See Also
    --------
    any : equivalent function; see for details.
    """
    return any(*args, **kwargs)