import re
from io import StringIO
import numba
from numba.core import types
from numba import jit, njit
from numba.tests.support import override_config, TestCase
import unittest
@TestCase.run_test_in_subprocess
def test_exercise_code_path(self):
    """
        Ensures template.html is available
        """

    def foo(n, a):
        s = a
        for i in range(n):
            s += i
        return s
    cfunc = njit((types.int32, types.int32))(foo)
    cres = cfunc.overloads[cfunc.signatures[0]]
    ta = cres.type_annotation
    buf = StringIO()
    ta.html_annotate(buf)
    output = buf.getvalue()
    buf.close()
    self.assertIn('foo', output)