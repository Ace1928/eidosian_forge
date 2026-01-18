import unittest
from numba import jit, njit, objmode, typeof, literally
from numba.extending import overload
from numba.core import types
from numba.core.errors import UnsupportedError
from numba.tests.support import (
def test_list_extend_forceobj(self):

    def consumer(*x):
        return x

    @jit(forceobj=True)
    def foo(x):
        return consumer(1, 2, *x)
    got = foo('ijo')
    expect = foo.py_func('ijo')
    self.assertEqual(got, (1, 2, 'i', 'j', 'o'))
    self.assertEqual(got, expect)