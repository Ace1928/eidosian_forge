import sys
import subprocess
import types as pytypes
import os.path
import numpy as np
import builtins
from numba.core import types
from numba.tests.support import TestCase, temp_directory
from numba.misc.help.inspector import inspect_function, inspect_module
def test_inspect_module(self):
    c = 0
    for it in inspect_module(builtins):
        self.assertIsInstance(it['module'], pytypes.ModuleType)
        self.assertIsInstance(it['name'], str)
        self.assertTrue(callable(it['obj']))
        self.check_function_descriptor(it)
        c += 1
    self.assertGreater(c, 0)