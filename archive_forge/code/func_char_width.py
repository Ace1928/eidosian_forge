from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
def char_width(self):
    _type_Py_UNICODE = gdb.lookup_type('Py_UNICODE')
    return _type_Py_UNICODE.sizeof