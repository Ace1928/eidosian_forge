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
def test_incref_after_cast(self):

    def f():
        return (0.0, np.zeros(1, dtype=np.int32))
    cfunc = njit(types.Tuple((types.complex128, types.Array(types.int32, 1, 'C')))())(f)
    z, arr = cfunc()
    self.assertPreciseEqual(z, 0j)
    self.assertPreciseEqual(arr, np.zeros(1, dtype=np.int32))