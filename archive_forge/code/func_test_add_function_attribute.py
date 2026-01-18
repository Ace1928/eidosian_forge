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
def test_add_function_attribute(self):
    mod = self.module()
    fn = mod.get_function('sum')
    fn.add_function_attribute('nocapture')
    with self.assertRaises(ValueError) as raises:
        fn.add_function_attribute('zext')
    self.assertEqual(str(raises.exception), "no such attribute 'zext'")