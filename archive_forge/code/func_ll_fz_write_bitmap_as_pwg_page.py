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
def ll_fz_write_bitmap_as_pwg_page(out, bitmap, pwg):
    """
    Low-level wrapper for `::fz_write_bitmap_as_pwg_page()`.
    Write a bitmap as a PWG page.

    Caller should provide a file header by calling
    fz_write_pwg_file_header, but can then write several pages to
    the same file.
    """
    return _mupdf.ll_fz_write_bitmap_as_pwg_page(out, bitmap, pwg)