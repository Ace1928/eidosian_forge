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
def test_get_process_triple(self):

    def normalize_ppc(arch):
        if arch == 'powerpc64le':
            return 'ppc64le'
        else:
            return arch
    triple = llvm.get_process_triple()
    default = llvm.get_default_triple()
    self.assertIsInstance(triple, str)
    self.assertTrue(triple)
    default_arch = normalize_ppc(default.split('-')[0])
    triple_arch = normalize_ppc(triple.split('-')[0])
    self.assertEqual(default_arch, triple_arch)