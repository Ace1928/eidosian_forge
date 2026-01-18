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
def for_CF_orders(name='order'):
    """Decorator that checks the fixture with orders 'C' and 'F'.

    Args:
         name(str): Argument name to which the specified order is passed.

    .. seealso:: :func:`cupy.testing.for_all_dtypes`

    """
    return for_orders([None, 'C', 'F', 'c', 'f'], name)