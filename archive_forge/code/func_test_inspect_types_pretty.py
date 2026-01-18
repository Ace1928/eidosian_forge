import multiprocessing
import platform
import threading
import pickle
import weakref
from itertools import chain
from io import StringIO
import numpy as np
from numba import njit, jit, typeof, vectorize
from numba.core import types, errors
from numba import _dispatcher
from numba.tests.support import TestCase, captured_stdout
from numba.np.numpy_support import as_dtype
from numba.core.dispatcher import Dispatcher
from numba.extending import overload
from numba.tests.support import needs_lapack, SerialMixin
from numba.testing.main import _TIMEOUT as _RUNNER_TIMEOUT
import unittest
@unittest.skipIf(jinja2 is None, "please install the 'jinja2' package")
@unittest.skipIf(pygments is None, "please install the 'pygments' package")
def test_inspect_types_pretty(self):

    @jit
    def foo(a, b):
        return a + b
    foo(1, 2)
    with captured_stdout():
        ann = foo.inspect_types(pretty=True)
    for k, v in ann.ann.items():
        span_found = False
        for line in v['pygments_lines']:
            if 'span' in line[2]:
                span_found = True
        self.assertTrue(span_found)
    with self.assertRaises(ValueError) as raises:
        foo.inspect_types(file=StringIO(), pretty=True)
    self.assertIn('`file` must be None if `pretty=True`', str(raises.exception))