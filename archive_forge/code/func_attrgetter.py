import array
import numpy as np
from numba import jit
from numba.tests.support import TestCase, compile_function, MemoryLeakMixin
import unittest
def attrgetter(attr):
    code = 'def func(x):\n        return x.%(attr)s\n' % locals()
    pyfunc = compile_function('func', code, globals())
    return jit(nopython=True)(pyfunc)