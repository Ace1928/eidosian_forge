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
def ll_fz_open_tar_archive_with_stream(file):
    """
    Low-level wrapper for `::fz_open_tar_archive_with_stream()`.
    Open a tar archive stream.

    Open an archive using a seekable stream object rather than
    opening a file or directory on disk.

    An exception is thrown if the stream is not a tar archive as
    indicated by the presence of a tar signature.

    """
    return _mupdf.ll_fz_open_tar_archive_with_stream(file)