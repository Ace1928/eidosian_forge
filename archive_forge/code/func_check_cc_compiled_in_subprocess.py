import contextlib
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from unittest import skip
from ctypes import *
import numpy as np
import llvmlite.binding as ll
from numba.core import utils
from numba.tests.support import (TestCase, tag, import_dynamic, temp_directory,
import unittest
def check_cc_compiled_in_subprocess(self, lib, code):
    prolog = "if 1:\n            import sys\n            import types\n            # to disable numba package\n            sys.modules['numba'] = types.ModuleType('numba')\n            try:\n                from numba import njit\n            except ImportError:\n                pass\n            else:\n                raise RuntimeError('cannot disable numba package')\n\n            sys.path.insert(0, %(path)r)\n            import %(name)s as lib\n            " % {'name': lib.__name__, 'path': os.path.dirname(lib.__file__)}
    code = prolog.strip(' ') + code
    subprocess.check_call([sys.executable, '-c', code])