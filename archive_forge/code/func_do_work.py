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
def do_work():
    tracking = []
    for c in range(50):
        tracking.append(llvm.JITLibraryBuilder().add_ir(llvm_ir).export_symbol('sum').link(lljit, f'sum_{i}_{c}'))