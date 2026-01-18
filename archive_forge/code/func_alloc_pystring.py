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
def alloc_pystring(self, string):
    stringp = self.alloc_string(string)
    PyString_FromStringAndSize = 'PyString_FromStringAndSize'
    try:
        gdb.parse_and_eval(PyString_FromStringAndSize)
    except RuntimeError:
        PyString_FromStringAndSize = 'PyUnicode%s_FromStringAndSize' % (get_inferior_unicode_postfix(),)
    try:
        result = gdb.parse_and_eval('(PyObject *) %s((char *) %d, (size_t) %d)' % (PyString_FromStringAndSize, stringp, len(string)))
    finally:
        self.free(stringp)
    pointer = pointervalue(result)
    if pointer == 0:
        raise gdb.GdbError('Unable to allocate Python string in the inferior.')
    return pointer