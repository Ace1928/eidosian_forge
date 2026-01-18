import unittest
from numba.tests.support import TestCase, run_in_subprocess
def test_laziness(self):
    """
        Importing top-level numba features should not import too many modules.
        """
    banlist = ['cffi', 'distutils', 'numba.cuda', 'numba.cpython.mathimpl', 'numba.cpython.randomimpl', 'numba.tests', 'numba.core.typing.collections', 'numba.core.typing.listdecl', 'numba.core.typing.npdatetime']
    for mod in banlist:
        if mod not in ('cffi',):
            __import__(mod)
    code = 'if 1:\n            from numba import jit, vectorize\n            from numba.core import types\n            import sys\n            print(list(sys.modules))\n            '
    out, _ = run_in_subprocess(code)
    modlist = set(eval(out.strip()))
    unexpected = set(banlist) & set(modlist)
    self.assertFalse(unexpected, 'some modules unexpectedly imported')