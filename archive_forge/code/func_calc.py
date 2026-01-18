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
@njit
def calc(base):
    tmp = 0
    for i in range(base.size):
        elem = base[i]
        tmp += elem.i1 * elem.f2 / elem.d3
        tmp += base[i].af4.sum()
    return tmp