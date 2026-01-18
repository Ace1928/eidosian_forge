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
def test_file_location(self):
    bar1 = self.import_bar1()
    bar2 = self.import_bar2()
    idxname1 = bar1._cache._cache_file._index_name
    idxname2 = bar2._cache._cache_file._index_name
    self.assertNotEqual(idxname1, idxname2)
    self.assertTrue(idxname1.startswith('__init__.bar-3.py'))
    self.assertTrue(idxname2.startswith('foo.bar-3.py'))