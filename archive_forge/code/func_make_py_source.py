import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def make_py_source(ffi, module_name, target_py_file, verbose=False):
    return _make_c_or_py_source(ffi, module_name, None, target_py_file, verbose)