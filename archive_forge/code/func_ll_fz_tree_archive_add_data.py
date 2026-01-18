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
def ll_fz_tree_archive_add_data(arch_, name, data, size):
    """
    Low-level wrapper for `::fz_tree_archive_add_data()`.
    Add a named block of data to an existing tree archive.

    The data will be copied into a buffer, and so the caller
    may free it as soon as this returns.
    """
    return _mupdf.ll_fz_tree_archive_add_data(arch_, name, data, size)