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
def test_get_section_content(self):
    elf = bytes.fromhex(issue_632_elf)
    obj = llvm.ObjectFileRef.from_data(elf)
    for s in obj.sections():
        if s.is_text():
            self.assertEqual(len(s.data()), 31)
            self.assertEqual(s.data().hex(), issue_632_text)