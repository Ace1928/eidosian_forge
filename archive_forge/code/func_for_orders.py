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
def for_orders(orders, name='order'):
    """Decorator to parameterize tests with order.

    Args:
         orders(list of order): orders to be tested.
         name(str): Argument name to which the specified order is passed.

    This decorator adds a keyword argument specified by ``name``
    to the test fixtures. Then, the fixtures run by passing each element of
    ``orders`` to the named argument.

    """

    def decorator(impl):

        @_wraps_partial(impl, name)
        def test_func(*args, **kw):
            for order in orders:
                try:
                    kw[name] = order
                    impl(*args, **kw)
                except Exception:
                    print(name, 'is', order)
                    raise
        return test_func
    return decorator