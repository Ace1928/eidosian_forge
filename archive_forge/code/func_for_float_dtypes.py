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
def for_float_dtypes(name='dtype', no_float16=False):
    """Decorator that checks the fixture with float dtypes.

    Args:
         name(str): Argument name to which specified dtypes are passed.
         no_float16(bool): If ``True``, ``numpy.float16`` is
             omitted from candidate dtypes.

    dtypes to be tested are ``numpy.float16`` (optional), ``numpy.float32``,
    and ``numpy.float64``.

    .. seealso:: :func:`cupy.testing.for_dtypes`,
        :func:`cupy.testing.for_all_dtypes`
    """
    if no_float16:
        return for_dtypes(_regular_float_dtypes, name=name)
    else:
        return for_dtypes(_float_dtypes, name=name)