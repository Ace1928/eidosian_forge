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
def numpy_cupy_array_list_equal(err_msg='', verbose=True, name='xp', sp_name=None, scipy_name=None):
    """Decorator that checks the resulting lists of NumPy and CuPy's one are equal.

    Args:
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values are appended
             to the error message.
         name(str): Argument name whose value is either
             ``numpy`` or ``cupy`` module.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``cupyx.scipy.sparse`` module. If ``None``, no
             argument is given for the modules.
         scipy_name(str or None): Argument name whose value is either ``scipy``
             or ``cupyx.scipy`` module. If ``None``, no argument is given for
             the modules.

    Decorated test fixture is required to return the same list of arrays
    (except the type of array module) even if ``xp`` is ``numpy`` or ``cupy``.

    .. seealso:: :func:`cupy.testing.assert_array_list_equal`
    """
    warnings.warn('numpy_cupy_array_list_equal is deprecated. Use numpy_cupy_array_equal instead.', DeprecationWarning)

    def check_func(x, y):
        _array.assert_array_equal(x, y, err_msg, verbose)
    return _make_decorator(check_func, name, False, False, False, sp_name, scipy_name)