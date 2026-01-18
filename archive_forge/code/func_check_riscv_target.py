import ctypes
import threading
from ctypes import CFUNCTYPE, c_int, c_int32
from ctypes.util import find_library
import gc
import locale
import os
import platform
import re
import subprocess
import sys
import unittest
from contextlib import contextmanager
from tempfile import mkstemp
from llvmlite import ir
from llvmlite import binding as llvm
from llvmlite.binding import ffi
from llvmlite.tests import TestCase
def check_riscv_target(self):
    try:
        llvm.Target.from_triple(self.triple)
    except RuntimeError as e:
        if 'No available targets are compatible with triple' in str(e):
            self.skipTest('RISCV target unsupported by linked LLVM.')
        else:
            raise e