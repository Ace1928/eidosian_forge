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
def for_all_dtypes_combination(names=('dtyes',), no_float16=False, no_bool=False, full=None, no_complex=False):
    """Decorator that checks the fixture with a product set of all dtypes.

    Args:
         names(list of str): Argument names to which dtypes are passed.
         no_float16(bool): If ``True``, ``numpy.float16`` is
             omitted from candidate dtypes.
         no_bool(bool): If ``True``, ``numpy.bool_`` is
             omitted from candidate dtypes.
         full(bool): If ``True``, then all combinations of dtypes
             will be tested.
             Otherwise, the subset of combinations will be tested
             (see description in :func:`cupy.testing.for_dtypes_combination`).
         no_complex(bool): If, True, ``numpy.complex64`` and
             ``numpy.complex128`` are omitted from candidate dtypes.

    .. seealso:: :func:`cupy.testing.for_dtypes_combination`
    """
    types = _make_all_dtypes(no_float16, no_bool, no_complex)
    return for_dtypes_combination(types, names, full)