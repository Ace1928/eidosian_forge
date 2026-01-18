from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
def ll_fz_hash_find(table, key):
    """
    Low-level wrapper for `::fz_hash_find()`.
    Search for a matching hash within the table, and return the
    associated value.
    """
    return _mupdf.ll_fz_hash_find(table, key)