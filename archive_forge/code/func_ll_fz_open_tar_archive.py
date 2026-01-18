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
def ll_fz_open_tar_archive(filename):
    """
    Low-level wrapper for `::fz_open_tar_archive()`.
    Open a tar archive file.

    An exception is thrown if the file is not a tar archive as
    indicated by the presence of a tar signature.

    filename: a path to a tar archive file as it would be given to
    open(2).
    """
    return _mupdf.ll_fz_open_tar_archive(filename)