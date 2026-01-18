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
def ll_fz_is_cfb_archive(file):
    """
    Low-level wrapper for `::fz_is_cfb_archive()`.
    Detect if stream object is a cfb archive.

    Assumes that the stream object is seekable.
    """
    return _mupdf.ll_fz_is_cfb_archive(file)