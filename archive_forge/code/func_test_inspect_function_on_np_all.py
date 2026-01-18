import sys
import subprocess
import types as pytypes
import os.path
import numpy as np
import builtins
from numba.core import types
from numba.tests.support import TestCase, temp_directory
from numba.misc.help.inspector import inspect_function, inspect_module
def test_inspect_function_on_np_all(self):
    info = inspect_function(np.all)
    self.check_function_descriptor(info, must_be_defined=True)
    source_infos = info['source_infos']
    self.assertGreater(len(source_infos), 0)
    c = 0
    for srcinfo in source_infos.values():
        self.assertIsInstance(srcinfo['kind'], str)
        self.assertIsInstance(srcinfo['name'], str)
        self.assertIsInstance(srcinfo['sig'], str)
        self.assertIsInstance(srcinfo['filename'], str)
        self.assertIsInstance(srcinfo['lines'], tuple)
        self.assertIn('docstring', srcinfo)
        c += 1
    self.assertEqual(c, len(source_infos))