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
def ll_fz_open_cfb_archive_with_stream(file):
    """
    Low-level wrapper for `::fz_open_cfb_archive_with_stream()`.
    Open a cfb file as an archive.

    Open an archive using a seekable stream object rather than
    opening a file or directory on disk.

    An exception is thrown if the file is not recognised as a chm.
    """
    return _mupdf.ll_fz_open_cfb_archive_with_stream(file)