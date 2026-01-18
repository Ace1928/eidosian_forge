import os
import platform
import re
import textwrap
import warnings
import numpy as np
from numba.tests.support import (TestCase, override_config, override_env_config,
from numba import jit, njit
from numba.core import types, compiler, utils
from numba.core.errors import NumbaPerformanceWarning
from numba import prange
from numba.experimental import jitclass
import unittest
def test_jitclass(self):
    with override_config('DISABLE_JIT', True):
        with forbid_codegen():
            SimpleJITClass = jitclass(simple_class_spec)(SimpleClass)
            obj = SimpleJITClass()
            self.assertPreciseEqual(obj.h, 5)
            cfunc = jit(nopython=True)(simple_class_user)
            self.assertPreciseEqual(cfunc(obj), 5)