import numpy as np
import sys
import traceback
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import (TestCase, expected_failure_py311,
import unittest
def check_against_python(self, exec_mode, pyfunc, cfunc, expected_error_class, *args):
    assert exec_mode in (force_pyobj_flags, no_pyobj_flags, no_pyobj_flags_w_nrt, no_gil_flags)
    with self.assertRaises(expected_error_class) as pyerr:
        pyfunc(*args)
    with self.assertRaises(expected_error_class) as jiterr:
        cfunc(*args)
    self.assertEqual(pyerr.exception.args, jiterr.exception.args)
    if isinstance(pyerr.exception, (UDEArgsToSuper, UDENoArgSuper)):
        self.assertTrue(pyerr.exception == jiterr.exception)
    if exec_mode is no_pyobj_flags:
        try:
            pyfunc(*args)
        except Exception:
            py_frames = traceback.format_exception(*sys.exc_info())
            expected_frames = py_frames[-2:]
        try:
            cfunc(*args)
        except Exception:
            c_frames = traceback.format_exception(*sys.exc_info())
            got_frames = c_frames[-2:]
        for expf, gotf in zip(expected_frames, got_frames):
            self.assertIn(gotf, expf)