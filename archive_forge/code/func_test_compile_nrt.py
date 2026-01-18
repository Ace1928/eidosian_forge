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
def test_compile_nrt(self):
    with self.check_cc_compiled(self._test_module.cc_nrt) as lib:
        self.assertPreciseEqual(lib.zero_scalar(1), 0.0)
        res = lib.zeros(3)
        self.assertEqual(list(res), [0, 0, 0])
        if has_blas:
            res = lib.vector_dot(4)
            self.assertPreciseEqual(res, 30.0)
        val = np.float64([2.0, 5.0, 1.0, 3.0, 4.0])
        res = lib.np_argsort(val)
        expected = np.argsort(val)
        self.assertPreciseEqual(res, expected)
        code = 'if 1:\n                from numpy.testing import assert_equal\n                from numpy import float64, argsort\n                res = lib.zero_scalar(1)\n                assert res == 0.0\n                res = lib.zeros(3)\n                assert list(res) == [0, 0, 0]\n                if %(has_blas)s:\n                    res = lib.vector_dot(4)\n                    assert res == 30.0\n                val = float64([2., 5., 1., 3., 4.])\n                res = lib.np_argsort(val)\n                expected = argsort(val)\n                assert_equal(res, expected)\n                ' % dict(has_blas=has_blas)
        self.check_cc_compiled_in_subprocess(lib, code)