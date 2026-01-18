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
def test_first_class_function(self):
    mod = self.import_module()
    f = mod.first_class_function_usecase
    self.assertEqual(f(mod.first_class_function_mul, 1), 1)
    self.assertEqual(f(mod.first_class_function_mul, 10), 100)
    self.assertEqual(f(mod.first_class_function_add, 1), 2)
    self.assertEqual(f(mod.first_class_function_add, 10), 20)
    self.check_pycache(7)