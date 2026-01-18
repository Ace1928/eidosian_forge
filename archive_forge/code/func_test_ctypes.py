import inspect
import llvmlite.binding as ll
import multiprocessing
import numpy as np
import os
import stat
import shutil
import subprocess
import sys
import traceback
import unittest
import warnings
from numba import njit
from numba.core import codegen
from numba.core.caching import _UserWideCacheLocator
from numba.core.errors import NumbaWarning
from numba.parfors import parfor
from numba.tests.support import (
from numba import njit
from numba import njit
from file2 import function2
from numba import njit
def test_ctypes(self):
    mod = self.import_module()
    for f in [mod.use_c_sin, mod.use_c_sin_nest1, mod.use_c_sin_nest2]:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', NumbaWarning)
            self.assertPreciseEqual(f(0.0), 0.0)
            self.check_pycache(0)
        self.assertEqual(len(w), 1)
        self.assertIn('Cannot cache compiled function "{}"'.format(f.__name__), str(w[0].message))