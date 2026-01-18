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
@unittest.skipUnless(hasattr(multiprocessing, 'get_context'), 'Test requires multiprocessing.get_context')
def test_no_collision(self):
    bar1 = self.import_bar1()
    bar2 = self.import_bar2()
    with capture_cache_log() as buf:
        res1 = bar1()
    cachelog = buf.getvalue()
    self.assertEqual(cachelog.count('index saved'), 1)
    self.assertEqual(cachelog.count('data saved'), 1)
    self.assertEqual(cachelog.count('index loaded'), 0)
    self.assertEqual(cachelog.count('data loaded'), 0)
    with capture_cache_log() as buf:
        res2 = bar2()
    cachelog = buf.getvalue()
    self.assertEqual(cachelog.count('index saved'), 1)
    self.assertEqual(cachelog.count('data saved'), 1)
    self.assertEqual(cachelog.count('index loaded'), 0)
    self.assertEqual(cachelog.count('data loaded'), 0)
    self.assertNotEqual(res1, res2)
    try:
        mp = multiprocessing.get_context('spawn')
    except ValueError:
        print('missing spawn context')
    q = mp.Queue()
    proc = mp.Process(target=cache_file_collision_tester, args=(q, self.tempdir, self.modname_bar1, self.modname_bar2))
    proc.start()
    log1 = q.get()
    got1 = q.get()
    log2 = q.get()
    got2 = q.get()
    proc.join()
    self.assertEqual(got1, res1)
    self.assertEqual(got2, res2)
    self.assertEqual(log1.count('index saved'), 0)
    self.assertEqual(log1.count('data saved'), 0)
    self.assertEqual(log1.count('index loaded'), 1)
    self.assertEqual(log1.count('data loaded'), 1)
    self.assertEqual(log2.count('index saved'), 0)
    self.assertEqual(log2.count('data saved'), 0)
    self.assertEqual(log2.count('index loaded'), 1)
    self.assertEqual(log2.count('data loaded'), 1)