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
def get_nrt_api_table(self):
    from cffi import FFI
    ffi = FFI()
    nrt_get_api = ffi.cast('void* (*)()', _nrt_python.c_helpers['get_api'])
    table = nrt_get_api()
    return table