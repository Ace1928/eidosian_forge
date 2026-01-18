import math
import os
import platform
import sys
import re
import numpy as np
from numba import njit
from numba.core import types
from numba.core.runtime import (
from numba.core.extending import intrinsic, include_path
from numba.core.typing import signature
from numba.core.imputils import impl_ret_untracked
from llvmlite import ir
import llvmlite.binding as llvm
from numba.core.unsafe.nrt import NRT_get_api
from numba.tests.support import (EnableNRTStatsMixin, TestCase, temp_directory,
from numba.core.registry import cpu_target
import unittest
def test_fake_memoryview(self):
    d = Dummy()
    self.assertEqual(Dummy.alive, 1)
    addr = 3735931646
    mi = rtsys.meminfo_new(addr, d)
    self.assertEqual(mi.refcount, 1)
    mview = memoryview(mi)
    self.assertEqual(mi.refcount, 1)
    self.assertEqual(addr, mi.data)
    self.assertFalse(mview.readonly)
    self.assertIs(mi, mview.obj)
    self.assertTrue(mview.c_contiguous)
    self.assertEqual(mview.itemsize, 1)
    self.assertEqual(mview.ndim, 1)
    del d
    del mi
    self.assertEqual(Dummy.alive, 1)
    del mview
    self.assertEqual(Dummy.alive, 0)