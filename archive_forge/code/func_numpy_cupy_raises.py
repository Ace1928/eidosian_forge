import functools
import inspect
import os
import random
from typing import Tuple, Type
import traceback
import unittest
import warnings
import numpy
import cupy
from cupy.testing import _array
from cupy.testing import _parameterized
import cupyx
import cupyx.scipy.sparse
from cupy.testing._pytest_impl import is_available
def numpy_cupy_raises(name='xp', sp_name=None, scipy_name=None, accept_error=Exception):
    """Decorator that checks the NumPy and CuPy throw same errors.

    Args:
         name(str): Argument name whose value is either
             ``numpy`` or ``cupy`` module.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``cupyx.scipy.sparse`` module. If ``None``, no
             argument is given for the modules.
         scipy_name(str or None): Argument name whose value is either ``scipy``
             or ``cupyx.scipy`` module. If ``None``, no argument is given for
             the modules.
         accept_error(bool, Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and CuPy test raises the
             same type of errors, and the type of the errors is specified with
             this argument, the errors are ignored and not raised.
             If it is ``True`` all error types are acceptable.
             If it is ``False`` no error is acceptable.

    Decorated test fixture is required throw same errors
    even if ``xp`` is ``numpy`` or ``cupy``.
    """
    warnings.warn('cupy.testing.numpy_cupy_raises is deprecated.', DeprecationWarning)

    def decorator(impl):

        @_wraps_partial_xp(impl, name, sp_name, scipy_name)
        def test_func(*args, **kw):
            cupy_result, cupy_error, numpy_result, numpy_error = _call_func_numpy_cupy(impl, args, kw, name, sp_name, scipy_name)
            _check_cupy_numpy_error(cupy_error, numpy_error, accept_error=accept_error)
        return test_func
    return decorator