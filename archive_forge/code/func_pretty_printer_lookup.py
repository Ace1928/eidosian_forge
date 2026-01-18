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
def pretty_printer_lookup(gdbval):
    type = gdbval.type.unqualified()
    if type.code != gdb.TYPE_CODE_PTR:
        return None
    type = type.target().unqualified()
    t = str(type)
    if t in ('PyObject', 'PyFrameObject', 'PyUnicodeObject', 'wrapperobject'):
        return PyObjectPtrPrinter(gdbval)