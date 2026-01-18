import cffi  # type: ignore
import os
import re
import sys
import warnings
import situation  # preloaded in setup.py
import importlib
def read_source(src_filename):
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'rinterface_lib', src_filename)) as fh:
        cdef = fh.read()
    return cdef