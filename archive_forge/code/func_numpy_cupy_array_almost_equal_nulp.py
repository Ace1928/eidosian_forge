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
def numpy_cupy_array_almost_equal_nulp(nulp=1, name='xp', type_check=True, accept_error=False, sp_name=None, scipy_name=None):
    """Decorator that checks results of NumPy and CuPy are equal w.r.t. spacing.

    Args:
         nulp(int): The maximum number of unit in the last place for tolerance.
         name(str): Argument name whose value is either
             ``numpy`` or ``cupy`` module.
         type_check(bool): If ``True``, consistency of dtype is also checked.
         accept_error(bool, Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and CuPy test raises the
             same type of errors, and the type of the errors is specified with
             this argument, the errors are ignored and not raised.
             If it is ``True``, all error types are acceptable.
             If it is ``False``, no error is acceptable.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``cupyx.scipy.sparse`` module. If ``None``, no
             argument is given for the modules.
         scipy_name(str or None): Argument name whose value is either ``scipy``
             or ``cupyx.scipy`` module. If ``None``, no argument is given for
             the modules.

    Decorated test fixture is required to return the same arrays
    in the sense of :func:`cupy.testing.assert_array_almost_equal_nulp`
    (except the type of array module) even if ``xp`` is ``numpy`` or ``cupy``.

    .. seealso:: :func:`cupy.testing.assert_array_almost_equal_nulp`
    """

    def check_func(x, y):
        _array.assert_array_almost_equal_nulp(x, y, nulp)
    return _make_decorator(check_func, name, type_check, False, accept_error, sp_name, scipy_name=None)