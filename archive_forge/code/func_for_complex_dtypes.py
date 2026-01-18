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
def for_complex_dtypes(name='dtype'):
    """Decorator that checks the fixture with complex dtypes.

    Args:
         name(str): Argument name to which specified dtypes are passed.

    dtypes to be tested are ``numpy.complex64`` and ``numpy.complex128``.

    .. seealso:: :func:`cupy.testing.for_dtypes`,
        :func:`cupy.testing.for_all_dtypes`
    """
    return for_dtypes(_complex_dtypes, name=name)