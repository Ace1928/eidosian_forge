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
def fz_close_zip_writer(self):
    """
        Class-aware wrapper for `::fz_close_zip_writer()`.
        	Close the zip file for writing.

        	This flushes any pending data to the file. This can throw
        	exceptions.
        """
    return _mupdf.FzZipWriter_fz_close_zip_writer(self)