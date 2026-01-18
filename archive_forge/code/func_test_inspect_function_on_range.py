import sys
import subprocess
import types as pytypes
import os.path
import numpy as np
import builtins
from numba.core import types
from numba.tests.support import TestCase, temp_directory
from numba.misc.help.inspector import inspect_function, inspect_module
def test_inspect_function_on_range(self):
    info = inspect_function(range)
    self.check_function_descriptor(info, must_be_defined=True)