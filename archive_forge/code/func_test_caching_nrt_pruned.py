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
def test_caching_nrt_pruned(self):
    self.check_pycache(0)
    mod = self.import_module()
    self.check_pycache(0)
    f = mod.add_usecase
    self.assertPreciseEqual(f(2, 3), 6)
    self.check_pycache(2)
    self.assertPreciseEqual(f(2, np.arange(3)), 2 + np.arange(3) + 1)
    self.check_pycache(3)
    self.check_hits(f, 0, 2)