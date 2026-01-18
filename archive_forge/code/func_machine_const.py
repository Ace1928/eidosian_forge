import contextlib
import ctypes
import struct
import sys
import llvmlite.ir as ir
import numpy as np
import unittest
from numba.core import types, typing, cgutils, cpu
from numba.core.compiler_lock import global_compiler_lock
from numba.tests.support import TestCase, run_in_subprocess
def machine_const(n):
    return ir.Constant(machine_int, n)