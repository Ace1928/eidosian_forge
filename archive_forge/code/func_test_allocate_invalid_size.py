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
@skip_if_32bit
def test_allocate_invalid_size(self):
    size = types.size_t.maxval // 8 // 2
    for pred in (True, False):
        with self.assertRaises(MemoryError) as raises:
            rtsys.meminfo_alloc(size, safe=pred)
        self.assertIn(f'Requested allocation of {size} bytes failed.', str(raises.exception))