import gc
import itertools
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
from numba.np import numpy_support
def check_argument_cleanup(self, typ, obj):
    """
        Check that argument cleanup doesn't leak references.
        """

    def f(x, y):
        pass

    def _objects(obj):
        objs = [obj]
        if isinstance(obj, tuple):
            for v in obj:
                objs += _objects(v)
        return objs
    objects = _objects(obj)
    cfunc = njit((typ, types.uint32))(f)
    with self.assertRefCount(*objects):
        cfunc(obj, 1)
    with self.assertRefCount(*objects):
        with self.assertRaises(OverflowError):
            cfunc(obj, -1)
    cfunc = njit((types.uint32, typ))(f)
    with self.assertRefCount(*objects):
        cfunc(1, obj)
    with self.assertRefCount(*objects):
        with self.assertRaises(OverflowError):
            cfunc(-1, obj)