import ctypes
import os
import subprocess
import sys
from collections import namedtuple
import numpy as np
from numba import cfunc, carray, farray, njit
from numba.core import types, typing, utils
import numba.core.typing.cffi_utils as cffi_support
from numba.tests.support import (TestCase, skip_unless_cffi, tag,
import unittest
from numba.np import numpy_support
def check_carray_usecase(self, pointer_factory, pyfunc, cfunc):
    expected = self.run_carray_usecase(pointer_factory, pyfunc)
    got = self.run_carray_usecase(pointer_factory, cfunc)
    self.assertPreciseEqual(expected, got)