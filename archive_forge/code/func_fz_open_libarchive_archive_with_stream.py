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
def fz_open_libarchive_archive_with_stream(self):
    """
        Class-aware wrapper for `::fz_open_libarchive_archive_with_stream()`.
        	Open an archive using libarchive.

        	Open an archive using a seekable stream object rather than
        	opening a file or directory on disk.

        	An exception is thrown if the stream is not supported by libarchive.
        """
    return _mupdf.FzStream_fz_open_libarchive_archive_with_stream(self)