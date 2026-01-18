from io import StringIO
import numpy as np
from numba.core import types
from numba.core.compiler import compile_extra, Flags
from numba.tests.support import TestCase, tag, MemoryLeakMixin
import unittest
def test_annotate_1(self):
    """
        Verify that annotation works as expected with one lifted loop
        """
    from numba import jit

    def bar():
        pass

    def foo(x):
        bar()
        for i in range(x.size):
            x[i] += 1
        return x
    cfoo = jit(forceobj=True)(foo)
    x = np.arange(10)
    xcopy = x.copy()
    r = cfoo(x)
    np.testing.assert_equal(r, xcopy + 1)
    buf = StringIO()
    cfoo.inspect_types(file=buf)
    annotation = buf.getvalue()
    buf.close()
    self.assertIn('The function contains lifted loops', annotation)
    line = foo.__code__.co_firstlineno + 2
    self.assertIn('Loop at line {line}'.format(line=line), annotation)
    self.assertIn('Has 1 overloads', annotation)