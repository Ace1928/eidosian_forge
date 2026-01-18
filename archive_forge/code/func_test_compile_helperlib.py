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
def test_compile_helperlib(self):
    with self.check_cc_compiled(self._test_module.cc_helperlib) as lib:
        res = lib.power(2, 7)
        self.assertPreciseEqual(res, 128)
        for val in (-1, -1 + 0j, np.complex128(-1)):
            res = lib.sqrt(val)
            self.assertPreciseEqual(res, 1j)
        for val in (4, 4.0, np.float64(4)):
            res = lib.np_sqrt(val)
            self.assertPreciseEqual(res, 2.0)
        res = lib.spacing(1.0)
        self.assertPreciseEqual(res, 2 ** (-52))
        self.assertNotEqual(lib.random(-1), lib.random(-1))
        res = lib.random(42)
        expected = np.random.RandomState(42).random_sample()
        self.assertPreciseEqual(res, expected)
        res = lib.size(np.float64([0] * 3))
        self.assertPreciseEqual(res, 3)
        code = 'if 1:\n                from numpy.testing import assert_equal, assert_allclose\n                res = lib.power(2, 7)\n                assert res == 128\n                res = lib.random(42)\n                assert_allclose(res, %(expected)s)\n                res = lib.spacing(1.0)\n                assert_allclose(res, 2**-52)\n                ' % {'expected': expected}
        self.check_cc_compiled_in_subprocess(lib, code)