import cffi  # type: ignore
import os
import re
import sys
import warnings
import situation  # preloaded in setup.py
import importlib
def create_cdef(definitions, header_filename):
    cdef = []
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'rinterface_lib', header_filename)) as fh:
        cdef, _ = c_preprocess(fh, definitions=definitions, rownum=0)
    return ''.join(cdef)