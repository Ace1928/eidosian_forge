import os
import platform
import re
import subprocess
import sys
import threading
from itertools import permutations
from numba import njit, gdb, gdb_init, gdb_breakpoint, prange
from numba.core import errors
from numba import jit
from numba.tests.support import (TestCase, captured_stdout, tag,
from numba.tests.gdb_support import needs_gdb
import unittest
def impl_gdb_call_w_bp(a):
    gdb_init('-ex', 'set confirm off', '-ex', 'c', '-ex', 'q')
    b = a + 1
    c = a * 2.34
    d = (a, b, c)
    gdb_breakpoint()
    print(a, b, c, d)