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
def fz_open_archive_with_stream(self):
    """
        Class-aware wrapper for `::fz_open_archive_with_stream()`.
        	Open zip or tar archive stream.

        	Open an archive using a seekable stream object rather than
        	opening a file or directory on disk.
        """
    return _mupdf.FzStream_fz_open_archive_with_stream(self)