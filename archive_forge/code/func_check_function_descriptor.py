import sys
import subprocess
import types as pytypes
import os.path
import numpy as np
import builtins
from numba.core import types
from numba.tests.support import TestCase, temp_directory
from numba.misc.help.inspector import inspect_function, inspect_module
def check_function_descriptor(self, info, must_be_defined=False):
    self.assertIsInstance(info, dict)
    self.assertIn('numba_type', info)
    numba_type = info['numba_type']
    if numba_type is None:
        self.assertFalse(must_be_defined)
    else:
        self.assertIsInstance(numba_type, types.Type)
        self.assertIn('explained', info)
        self.assertIsInstance(info['explained'], str)
        self.assertIn('source_infos', info)
        self.assertIsInstance(info['source_infos'], dict)