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
@skip_parfors_unsupported
@needs_gdb_harness
def test_gdb_split_init_and_break_w_parallel_nopython_impl(self):
    with captured_stdout():
        _dbg_njit(impl_gdb_split_init_and_break_w_parallel)(10)