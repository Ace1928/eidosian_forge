import re
from io import StringIO
import numba
from numba.core import types
from numba import jit, njit
from numba.tests.support import override_config, TestCase
import unittest
@TestCase.run_test_in_subprocess
def test_exercise_code_path_with_lifted_loop(self):
    """
        Ensures that lifted loops are handled correctly in obj mode
        """

    def bar(x):
        return x

    def foo(x):
        h = 0.0
        for i in range(x):
            h = h + i
        for k in range(x):
            h = h + k
        if x:
            h = h - bar(x)
        return h
    cfunc = jit((types.intp,), forceobj=True, looplift=True)(foo)
    cres = cfunc.overloads[cfunc.signatures[0]]
    ta = cres.type_annotation
    buf = StringIO()
    ta.html_annotate(buf)
    output = buf.getvalue()
    buf.close()
    self.assertIn('bar', output)
    self.assertIn('foo', output)
    self.assertIn('LiftedLoop', output)