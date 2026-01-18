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
def ll_fz_new_zip_writer(filename):
    """
    Low-level wrapper for `::fz_new_zip_writer()`.
    Create a new zip writer that writes to a given file.

    Open an archive using a seekable stream object rather than
    opening a file or directory on disk.
    """
    return _mupdf.ll_fz_new_zip_writer(filename)