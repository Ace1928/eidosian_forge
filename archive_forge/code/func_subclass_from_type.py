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
@classmethod
def subclass_from_type(cls, t):
    """
        Given a PyTypeObjectPtr instance wrapping a gdb.Value that's a
        (PyTypeObject*), determine the corresponding subclass of PyObjectPtr
        to use

        Ideally, we would look up the symbols for the global types, but that
        isn't working yet:
          (gdb) python print gdb.lookup_symbol('PyList_Type')[0].value
          Traceback (most recent call last):
            File "<string>", line 1, in <module>
          NotImplementedError: Symbol type not yet supported in Python scripts.
          Error while executing Python code.

        For now, we use tp_flags, after doing some string comparisons on the
        tp_name for some special-cases that don't seem to be visible through
        flags
        """
    try:
        tp_name = t.field('tp_name').string()
        tp_flags = int(t.field('tp_flags'))
    except (RuntimeError, UnicodeDecodeError):
        return cls
    name_map = {'bool': PyBoolObjectPtr, 'classobj': PyClassObjectPtr, 'NoneType': PyNoneStructPtr, 'frame': PyFrameObjectPtr, 'set': PySetObjectPtr, 'frozenset': PySetObjectPtr, 'builtin_function_or_method': PyCFunctionObjectPtr, 'method-wrapper': wrapperobject}
    if tp_name in name_map:
        return name_map[tp_name]
    if tp_flags & Py_TPFLAGS_HEAPTYPE:
        return HeapTypeObjectPtr
    if tp_flags & Py_TPFLAGS_LONG_SUBCLASS:
        return PyLongObjectPtr
    if tp_flags & Py_TPFLAGS_LIST_SUBCLASS:
        return PyListObjectPtr
    if tp_flags & Py_TPFLAGS_TUPLE_SUBCLASS:
        return PyTupleObjectPtr
    if tp_flags & Py_TPFLAGS_BYTES_SUBCLASS:
        return PyBytesObjectPtr
    if tp_flags & Py_TPFLAGS_UNICODE_SUBCLASS:
        return PyUnicodeObjectPtr
    if tp_flags & Py_TPFLAGS_DICT_SUBCLASS:
        return PyDictObjectPtr
    if tp_flags & Py_TPFLAGS_BASE_EXC_SUBCLASS:
        return PyBaseExceptionObjectPtr
    return cls