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
def for_dtypes(dtypes, name='dtype'):
    """Decorator for parameterized dtype test.

    Args:
         dtypes(list of dtypes): dtypes to be tested.
         name(str): Argument name to which specified dtypes are passed.

    This decorator adds a keyword argument specified by ``name``
    to the test fixture. Then, it runs the fixtures in parallel
    by passing the each element of ``dtypes`` to the named
    argument.
    """

    def decorator(impl):

        @_wraps_partial(impl, name)
        def test_func(*args, **kw):
            for dtype in dtypes:
                try:
                    kw[name] = numpy.dtype(dtype).type
                    impl(*args, **kw)
                except _skip_classes as e:
                    print('skipped: {} = {} ({})'.format(name, dtype, e))
                except Exception:
                    print(name, 'is', dtype)
                    raise
        return test_func
    return decorator