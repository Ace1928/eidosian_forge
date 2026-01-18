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
def for_unsigned_dtypes(name='dtype'):
    """Decorator that checks the fixture with unsinged dtypes.

    Args:
         name(str): Argument name to which specified dtypes are passed.

    dtypes to be tested are ``numpy.dtype('B')``, ``numpy.dtype('H')``,

     ``numpy.dtype('I')``, ``numpy.dtype('L')``, and ``numpy.dtype('Q')``.

    .. seealso:: :func:`cupy.testing.for_dtypes`,
        :func:`cupy.testing.for_all_dtypes`
    """
    return for_dtypes(_unsigned_dtypes, name=name)